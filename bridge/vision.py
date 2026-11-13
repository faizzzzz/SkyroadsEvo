"""Turn a game screenshot into the simulator's "vision" observation.

Calibration (bridge/calibration.json, made by calibrate.py) holds:
  road_quad:  4 screen points [near-left, near-right, far-right, far-left] of the
              road surface, in window-client pixels
  near_rows / far_rows: how many rows ahead of the ship those two edges are
  background: list of BGR colours that mean "nothing here" (space / void)
  ship:       BGR colour of the ship, plus the screen band (y0, y1) it lives in
  bg_tol, ship_tol: colour distance tolerances

Assumptions (check them in the debug window of play_real.py):
  * the road plane is drawn with a fixed camera relative to the road edges,
  * the road is LANES lanes wide between the quad's left and right edges,
  * raised blocks are told apart from floor only via the optional "raised" colours.
"""
import json

import cv2
import numpy as np

from starlane import config as C
from starlane.sim import LOOK_LANES, OBS_ROWS, clear_path_features, reduce_bands


class Vision:
    def __init__(self, calib_path):
        with open(calib_path) as f:
            self.cal = json.load(f)
        q = np.array(self.cal["road_quad"], np.float32)
        nr, fr = float(self.cal["near_rows"]), float(self.cal["far_rows"])
        # road plane (lane x, rows ahead) -> screen
        plane = np.array([[0, nr], [C.LANES, nr], [C.LANES, fr], [0, fr]], np.float32)
        self.H = cv2.getPerspectiveTransform(plane, q)
        self.bg = np.array(self.cal["background"], np.float32).reshape(-1, 3)
        self.raised = np.array(self.cal.get("raised", []), np.float32).reshape(-1, 3)
        self.ship_col = np.array(self.cal["ship"]["color"], np.float32)
        self.ship_band = self.cal["ship"]["band"]
        self.bg_tol = float(self.cal.get("bg_tol", 40))
        self.ship_tol = float(self.cal.get("ship_tol", 40))
        self.far_rows = fr
        # palette mode: listed road colours are floor, anything else is a hole
        self.floor = np.array(self.cal.get("floor", []), np.float32).reshape(-1, 3)
        self.ship_ref_y = self.cal.get("ship_ref_y")
        self.ship_box = None
        self.near_rows = nr
        self.last_x = C.LANES / 2

    def plane_to_screen(self, xs, ds):
        pts = np.stack([xs.ravel(), ds.ravel()], 1).astype(np.float32)[None]
        return cv2.perspectiveTransform(pts, self.H)[0].reshape(*np.shape(xs), 2)

    def ship_x(self, img):
        """Estimate the ship's lateral road position from its colour."""
        y0, y1 = self.ship_band
        band = img[y0:y1].astype(np.float32)
        d = np.linalg.norm(band - self.ship_col, axis=2)
        ys, xs = np.nonzero(d < self.ship_tol)
        if len(xs) < 5:
            return self.last_x
        cx, cy = float(np.median(xs)), float(np.median(ys) + y0)
        self.ship_box = (cx, cy)
        if self.ship_ref_y is not None:   # map from road level, even mid-jump
            cy = float(self.ship_ref_y)
        # invert the homography at the ship's screen position
        inv = np.linalg.inv(self.H)
        p = inv @ np.array([cx, cy, 1.0])
        self.last_x = float(np.clip(p[0] / p[2], 0.01, C.LANES - 0.01))
        return self.last_x

    def cells(self, img, ship_x):
        """(bands, lanes) array: -1 hole, 0 floor, 1 raised; plus per-row sample pixels.
        Samples one point per road row (0..20 rows ahead) and lane, then groups rows
        into the same distance bands as the simulator."""
        lane_c = np.floor(ship_x) + 0.5
        rows = np.arange(OBS_ROWS, dtype=np.float32)
        xs = lane_c + LOOK_LANES[None, :] + 0 * rows[:, None]
        ds = rows[:, None] + 0 * LOOK_LANES[None, :]
        ds = np.clip(ds, self.near_rows, self.far_rows)
        px = self.plane_to_screen(xs, ds)
        h, w = img.shape[:2]
        ix = np.clip(px[..., 0].round().astype(int), 0, w - 1)
        iy = np.clip(px[..., 1].round().astype(int), 0, h - 1)
        col = img[iy, ix].astype(np.float32)                       # (R, L, 3)
        if len(self.floor):
            is_bg = np.min(np.linalg.norm(col[:, :, None, :] - self.floor[None, None], axis=3), 2) >= self.bg_tol
        else:
            is_bg = np.min(np.linalg.norm(col[:, :, None, :] - self.bg[None, None], axis=3), 2) < self.bg_tol
        out = np.where(is_bg, -1.0, 0.0)
        if len(self.raised):
            is_r = np.min(np.linalg.norm(col[:, :, None, :] - self.raised[None, None], axis=3), 2) < self.bg_tol
            out = np.where(is_r, 1.0, out) if len(self.floor) else np.where(is_r & ~is_bg, 1.0, out)
        if self.ship_box is not None:     # the ship's own sprite hides the road under it
            cx, cy = self.ship_box
            out = np.where((np.abs(ix - cx) < 85) & (np.abs(iy - cy) < 60), 0.0, out)
        off_road = (xs < 0) | (xs >= C.LANES)
        out = np.where(off_road, -1.0, out)
        return reduce_bands(out.astype(np.float32), axis=0), np.stack([ix, iy], -1), out

    def observe(self, img, est_speed, since_jump):
        x = self.ship_x(img)
        cells, pix, raw = self.cells(img, x)
        lane_c = np.floor(x) + 0.5
        state = np.array([(x - lane_c) * 2, x / C.LANES * 2 - 1, est_speed / C.V_MAX,
                          min(since_jump, 1.0)], np.float32)
        clear = clear_path_features(raw != 0)
        return np.concatenate([cells.ravel(), clear, state])[None], raw, pix, x
