"""Drive real game windows for evolution: read the screen, fly one attempt, report progress.

Everything works from pixels and key presses on a 640x480 DOSBox window (see
game/skyroads_train.conf). Nothing reads or changes the game's own files.

Screen facts this relies on, measured from the running game:
  * the gauge at the top left of the dashboard fills as the ship moves along the road
  * the outer ring of the centre dial shows speed
  * Esc returns to the road menu, Enter on a road starts it
  * the camera stays fixed to the road edges
"""
import ctypes
import json
import os
import struct
import subprocess
import threading
import time
from collections import deque
from ctypes import wintypes

import cv2
import numpy as np

from starlane import config as C
from starlane.sim import (LOOK_LANES, OBS_ROWS, clear_path_features, reduce_bands)
from bridge.memory import EmulatedRAM
from starlane.realobs import JumpTap, Steering, build_obs, lane_of
from bridge.winapi import (EnumWindowsProc, PostKeys, SCAN, WindowGrabber, client_rect, user32)

HERE = os.path.dirname(__file__)
GAME_DIR = os.path.join(HERE, "..", "game")
W, H = 640, 480

# road plane: lines converge at y=82.5; 7 lanes span this half width at each screen row
VANISH_Y, HALF_SLOPE = 82.5, 1.8445
NEAR_Y, FAR_Y = 240.0, 107.0          # 0 and 20 rows ahead of the ship
SHIP_GROUND_Y = 228

BAR = (332, 352, 82, 145)             # progress gauge: y0, y1, x0, x1
BAR_DARK, BAR_LIT = np.array([62, 7, 54]), np.array([139, 23, 109])

SHIP_BLUE = np.array([[177, 108, 50], [206, 149, 90], [225, 175, 114], [158, 84, 35]], np.int16)
SHIP_RED = np.array([[0, 18, 194]], np.int16)
SHIP_BAND = (100, 307)

ROADS = 30                            # 10 worlds x 3 roads, laid out in two columns of 5 worlds
PALETTE_FILE = os.path.join(HERE, "road_palettes.json")

_yy, _xx = np.mgrid[320:480, 240:410]
_r = np.hypot(_xx - 325, _yy - 407.5)
DIAL_RING = (_r > 52) & (_r < 72) & (_yy < 407)
DIAL_FULL = 0.98


# Where the running game keeps its state, found by correlating RAM with the screen.
MEM_SHIP = 0x8F38              # u16: forward fraction, forward row, lateral, height
MEM_SPEED = 0xC2D9             # u16
MEM_ROAD = 0x9748              # rows of 7 two-byte cells, indexed by forward row
BELOW_ROAD = 300               # deeper than this under road level the ship is falling and can't recover
ROW_BYTES = 14
GROUND = 10240                 # height at road level
from starlane.realobs import X_LEFT, X_LANE   # the game's lane geometry (shared with the inputs)
SPEED_FULL = 8704
CELL_TUNNEL = 1                # high byte of a road cell: 0 none, 1 tunnel, others are blocks
TILE_DEADLY = 0x0C             # low byte of a road cell: touching this at road level destroys the ship
CELL_LOW_BLOCK = 2             # (the road's last cell is a tunnel; 2, 3, 4 are not yet told apart for certain)
LOW_BLOCK_CLEAR = 2600         # height above road level that clears a low block
REFERENCE_JUMP = 5195          # peak jump height (above road level) at the gauge's 500 gravity
HANDOVER_ROWS = 12             # practice flights: the candidate takes over this far from the end
APPROACH_ROWS = 5              # rows before the last tunnel that must be driven, not hopped
TUNNEL_CODE = 0.5              # how a tunnel appears in the pilot's grid: not floor, not a block
MAX_ROWS = 400


class WindowLost(Exception):
    pass


def _near(img, colours, tol):
    """Boolean mask of pixels within tol (sum of channel differences) of any colour."""
    d = np.abs(img[..., None, :].astype(np.int16) - colours).sum(-1)
    return (d <= tol).any(-1)


class Palettes:
    """Per-road floor colours, learned from what the ship is seen driving on."""

    NEEDED = 6      # sightings before a colour counts as floor

    def __init__(self, path=PALETTE_FILE):
        self.path = path
        self.lock = threading.Lock()
        self.counts: dict[int, dict[tuple, int]] = {}
        if os.path.exists(path):
            with open(path) as f:
                for road, entries in json.load(f).items():
                    self.counts[int(road)] = {tuple(c): n for c, n in entries}

    def floors(self, road: int) -> np.ndarray:
        with self.lock:
            cols = [c for c, n in self.counts.get(road, {}).items() if n >= self.NEEDED]
        return np.array(cols, np.int16).reshape(-1, 3)

    def see(self, road: int, colour):
        with self.lock:
            d = self.counts.setdefault(road, {})
            key = tuple(int(v) for v in colour)
            d[key] = d.get(key, 0) + 1

    def add(self, road: int, colour: tuple, n: int):
        with self.lock:
            d = self.counts.setdefault(road, {})
            d[colour] = d.get(colour, 0) + n

    def save(self):
        with self.lock:
            data = {str(r): [[list(c), n] for c, n in d.items()] for r, d in self.counts.items()}
        with open(self.path, "w") as f:
            json.dump(data, f)


class RoadVision:
    """Builds the simulator's 'vision' observation from a 640x480 frame."""

    def __init__(self, palettes: Palettes):
        self.palettes = palettes
        def half(y):
            return HALF_SLOPE * (y - VANISH_Y)
        cx = W / 2
        quad = np.array([[cx - half(NEAR_Y), NEAR_Y], [cx + half(NEAR_Y), NEAR_Y],
                         [cx + half(FAR_Y), FAR_Y], [cx - half(FAR_Y), FAR_Y]], np.float32)
        plane = np.array([[0, 0], [C.LANES, 0], [C.LANES, 20], [0, 20]], np.float32)
        self.H = cv2.getPerspectiveTransform(plane, quad)
        self.Hinv = np.linalg.inv(self.H)
        self.samples = {}
        rows = np.arange(OBS_ROWS, dtype=np.float32)
        for lane in range(C.LANES):
            xs = lane + 0.5 + LOOK_LANES[None, :] + 0 * rows[:, None]
            ds = rows[:, None] + 0 * LOOK_LANES[None, :]
            pts = np.stack([xs.ravel(), ds.ravel()], 1).astype(np.float32)[None]
            p = cv2.perspectiveTransform(pts, self.H)[0].reshape(*xs.shape, 2)
            ix = np.clip(p[..., 0].round().astype(int), 2, W - 3)
            iy = np.clip(p[..., 1].round().astype(int), 2, H - 3)
            self.samples[lane] = (ix, iy, (xs < 0) | (xs >= C.LANES))

    @staticmethod
    def find_ship(img):
        y0, y1 = SHIP_BAND
        band = img[y0:y1:2, ::2]                # every other pixel is plenty for a sprite this size
        ys, xs = np.nonzero(_near(band, SHIP_BLUE, 8))
        if len(xs) < 7:
            return None
        cx, cy = float(np.median(xs)) * 2, float(np.median(ys)) * 2 + y0
        region = img[max(0, int(cy) - 25):int(cy) + 25, max(0, int(cx) - 50):int(cx) + 50]
        if _near(region, SHIP_RED, 8).sum() < 4:
            return None
        return cx, cy

    def ship_lane_x(self, cx):
        p = self.Hinv @ np.array([cx, NEAR_Y, 1.0])
        return float(np.clip(p[0] / p[2], 0.01, C.LANES - 0.01))

    def learn_floor(self, img, road, ship):
        """Colours just behind a ship that is holding its height are something it can ride on,
        whether that is the road or the top of a tunnel."""
        cx, cy = ship
        y = int(cy + 19)
        for x in (int(cx - 26), int(cx + 26)):
            if 3 <= x < W - 3 and y < H - 3:
                c = self.flat_colour(img, x, y)
                if c is not None:
                    self.palettes.see(road, c)

    @staticmethod
    def flat_colour(img, x, y):
        """The colour at (x, y) if it is a flat, non-dark patch, else None."""
        c = img[y, x]
        patch = img[y - 2:y + 3, x - 2:x + 3].astype(np.int16)
        if np.abs(patch - c.astype(np.int16)).sum(-1).max() <= 6 and int(c.sum()) > 60:
            return tuple(int(v) for v in c)
        return None

    def observe(self, img, road, ship, ship_x, speed, since_jump):
        lane = int(np.clip(np.floor(ship_x), 0, C.LANES - 1))
        ix, iy, off_road = self.samples[lane]
        centre = img[iy, ix].astype(np.int16)
        flat = np.ones_like(off_road)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            flat &= np.abs(img[iy + dy, ix + dx].astype(np.int16) - centre).sum(-1) <= 12
        code = np.full(off_road.shape, -1.0, np.float32)
        code[flat & (centre.sum(-1) > 60)] = 1.0       # some flat surface...
        floors = self.palettes.floors(road)
        if len(floors):                                 # ...floor, if we've driven on that colour
            is_floor = (np.abs(centre[..., None, :] - floors).sum(-1) <= 12).any(-1)
            code[is_floor & flat] = 0.0
        if ship is not None:                            # the ship hides the road under it
            code[(np.abs(ix - ship[0]) < 43) & (np.abs(iy - ship[1]) < 30)] = 0.0
        code[off_road] = -1.0
        lane_c = lane + 0.5
        state = np.array([(ship_x - lane_c) * 2, ship_x / C.LANES * 2 - 1, speed,
                          min(since_jump, 1.0)], np.float32)
        obs = np.concatenate([reduce_bands(code, axis=0).ravel(),
                              clear_path_features(code != 0), state])
        return obs.astype(np.float32).tolist(), code


def read_bar(img):
    y0, y1, x0, x1 = BAR
    bar = img[y0:y1, x0:x1].astype(np.int16)
    dark = int((np.abs(bar - BAR_DARK).sum(-1) <= 12).sum())
    lit = int((np.abs(bar - BAR_LIT).sum(-1) <= 12).sum())
    return dark, lit


def read_speed(img):
    d = img[320:480, 240:410].astype(np.int16)
    purple = (d[..., 2] - d[..., 1] > 40) & (d[..., 0] - d[..., 1] > 40)
    return float(min(1.0, purple[DIAL_RING].mean() / DIAL_FULL))


def slot_rect(road):
    world, r = divmod(road, 3)
    col, row = divmod(world, 5)
    y = 40 + row * 93.5 + r * 21.5
    x0, x1 = (125, 218) if col == 0 else (445, 540)
    return int(y - 11), int(y + 11), x0, x1


def read_cursor(img):
    """Which road the menu cursor is on, or None if this isn't the road menu."""
    white = img.min(-1) > 150
    scores = []
    for road in range(ROADS):
        y0, y1, x0, x1 = slot_rect(road)
        scores.append(int(white[y0:y1, x0:x1].sum()))
    best = int(np.argmax(scores))
    others = sorted(scores)[-2]
    return best if scores[best] >= 60 and scores[best] > 2 * others + 20 else None


def find_pid_window(pid):
    found = []
    def cb(hwnd, _):
        p = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        if p.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True
    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return found[0] if found else None


class _MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]


def second_monitor():
    """Work area (left, top, right, bottom) of a non-primary monitor, or None."""
    found = []
    def cb(hmon, _dc, _rect, _lp):
        info = _MONITORINFO()
        info.cbSize = ctypes.sizeof(info)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        if not info.dwFlags & 1:
            r = info.rcWork
            found.append((r.left, r.top, r.right, r.bottom))
        return True
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFO)]
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                              ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)(cb)
    user32.EnumDisplayMonitors(None, None, proc, 0)
    return found[0] if found else None


class GameWindow:
    """One private copy of the running game."""

    def __init__(self, index: int, vision: RoadVision, total: int = 1, conf: str = "", fast: bool = False):
        """fast: run the game unthrottled by holding DOSBox's speedlock key (F11, bound in
        game/skyroads_fast.map). Pilots still decide once per game update, so they fly the same way."""
        self.index = index
        self.fast = fast
        self.pace = 0.3 if fast else 1.0     # menu waits shrink when the game itself runs faster
        conf = conf or ("skyroads_fast.conf" if fast else "skyroads_train.conf")
        self.vision = vision
        exe = os.path.abspath(os.path.join(GAME_DIR, "dosbox", "dosbox.exe"))
        self.proc = subprocess.Popen([exe, "-conf", conf],
                                     cwd=os.path.abspath(GAME_DIR),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.hwnd = None
        t0 = time.time()
        while time.time() - t0 < 20 and not self.hwnd:
            self.hwnd = find_pid_window(self.proc.pid)
            time.sleep(0.2)
        if not self.hwnd:
            raise WindowLost("window never appeared")
        time.sleep(1.5)
        mon = second_monitor()
        if mon:                              # tile on the other monitor, out of the user's way
            # stay fully on screen (off-screen parts aren't drawn); overlapping is fine
            cols = min(4, total)
            rows = (total + cols - 1) // cols
            row, col = divmod(index, cols)
            step_x = (mon[2] - mon[0] - 660) // max(1, cols - 1)
            step_y = (mon[3] - mon[1] - 525) // max(1, rows - 1)
            user32.SetWindowPos(self.hwnd, None, mon[0] + col * min(step_x, 660),
                                mon[1] + row * min(step_y, 525), 0, 0,
                                0x0001 | 0x0004 | 0x0010)   # NOSIZE | NOZORDER | NOACTIVATE
            time.sleep(0.3)
        self.resizes = 0
        self.road = None                     # road the game is currently on
        self.gravity_cache = {}              # road -> measured gravity input
        self.restarting = False              # the game will restart that road by itself
        self.faded = False
        self.ram = EmulatedRAM(self.proc.pid)
        self.grabber = WindowGrabber(self.hwnd)
        self.keys = PostKeys(self.hwnd, list(SCAN))
        if fast:
            self.keys.set("f11", True)

    def alive(self):
        return self.proc.poll() is None

    def fix_size(self):
        """Put the window back to 640x480 if something resized it, without taking focus."""
        self.resizes += 1
        user32.ShowWindow(self.hwnd, 4)         # SW_SHOWNOACTIVATE: also un-maximizes
        time.sleep(0.2)
        outer = wintypes.RECT()
        user32.GetWindowRect(self.hwnd, ctypes.byref(outer))
        cw, ch = client_rect(self.hwnd)[2:]
        user32.SetWindowPos(self.hwnd, None, 0, 0,
                            outer.right - outer.left + W - cw, outer.bottom - outer.top + H - ch,
                            0x0002 | 0x0004 | 0x0010)   # NOMOVE | NOZORDER | NOACTIVATE
        time.sleep(0.4)

    def grab(self):
        if not self.alive():
            raise WindowLost("game closed")
        for _ in range(4):
            if client_rect(self.hwnd)[2:] == (W, H):
                return self.grabber.grab()
            self.fix_size()
        raise WindowLost(f"window stuck at {client_rect(self.hwnd)[2:]}")

    def tap(self, name, hold=0.08):
        self.keys.set(name, True)
        time.sleep(hold)
        self.keys.set(name, False)

    def to_menu(self, timeout=30.0):
        """Reach the road menu from whatever is on screen."""
        t0 = time.time()
        last = 0.0
        while time.time() - t0 < timeout:
            img = self.grab()
            if read_cursor(img) is not None:
                time.sleep(0.5 * self.pace)     # keys are ignored during the fade-in
                if read_cursor(self.grab()) is not None:
                    return
            dark, lit = read_bar(img)
            if img.mean() < 40:                 # fading: a key now could start whatever is highlighted
                time.sleep(0.1 * self.pace)
                continue
            if time.time() - last > 1.2 * self.pace:
                self.tap("esc" if dark + lit > 300 else "enter")
                last = time.time()
            time.sleep(0.1 * self.pace)
        raise WindowLost("could not reach the road menu")

    def select(self, road, timeout=12.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            img = self.grab()
            cur = read_cursor(img)
            if cur == road:
                return
            if cur is None:
                dark, lit = read_bar(img)
                if dark + lit > 300:            # a road started by accident: back to the menu
                    self.tap("esc")
                    time.sleep(1.5 * self.pace)
                    continue
            if cur is not None:
                cw, cr = divmod(cur, 3)
                tw, tr = divmod(road, 3)
                ccol, crow = cw // 5, (cw % 5) * 3 + cr
                tcol, trow = tw // 5, (tw % 5) * 3 + tr
                if ccol != tcol:
                    self.tap("right" if tcol > ccol else "left")
                else:
                    self.tap("down" if trow > crow else "up")
                # wait for the press to show before deciding the next one: on a busy PC the menu
                # reacts late, and pressing again on a stale picture overshoots back and forth
                moved_by = time.time() + 1.0
                while time.time() < moved_by:
                    time.sleep(0.03)
                    now = read_cursor(self.grab())
                    if now is not None and now != cur:
                        break
                continue
            time.sleep(0.2 * self.pace)
        raise WindowLost(f"could not select road {road}")

    def start_race(self, timeout=15.0):
        self.tap("enter")
        t0 = time.time()
        while time.time() - t0 < timeout:
            img = self.grab()
            dark, lit = read_bar(img)
            if dark + lit > 300 and RoadVision.find_ship(img) is not None:
                time.sleep(0.15 * self.pace)
                return
            time.sleep(0.05)
        raise WindowLost("race did not start")

    def await_restart(self, timeout=9.0):
        """After a lost ship the game restarts the road itself; wait for that."""
        t0 = time.time()
        faded = self.faded
        while time.time() - t0 < timeout:
            img = self.grab()
            dark, lit = read_bar(img)
            if dark + lit <= 300:
                faded = True
            elif faded and lit < 0.05 * (dark + lit) and RoadVision.find_ship(img) is not None:
                time.sleep(0.15)
                return True
            time.sleep(0.03)
        return False

    def read_ship(self):
        frac, row, x, y = struct.unpack("<4H", self.ram.read(MEM_SHIP, 8))
        speed = struct.unpack("<H", self.ram.read(MEM_SPEED, 2))[0]
        return row, frac / 65536, x, y, speed

    def road_length(self):
        """Rows in the current road: where a long run of empty rows begins."""
        cells = np.frombuffer(self.ram.read(MEM_ROAD, MAX_ROWS * ROW_BYTES), np.uint16).reshape(-1, 7)
        blank = (cells == 0).all(1)
        for i in range(3, MAX_ROWS - 25):
            if blank[i:i + 25].all():
                return i
        return MAX_ROWS

    def measure_gravity(self, road):
        """Roads differ in gravity. One standing jump measures it; the ship doesn't move forward."""
        if road not in self.gravity_cache:
            if self.fast:                        # a jump is too quick to sample when fast-forwarded
                self.keys.set("f11", False)
                time.sleep(0.1)
            self.keys.set("space", True)
            t0, peak = time.time(), GROUND
            while time.time() - t0 < 1.6:
                y = self.read_ship()[3]
                peak = max(peak, y)
                if time.time() - t0 > 0.2 and y <= GROUND:
                    break
                time.sleep(0.008)
            self.keys.set("space", False)
            time.sleep(0.12)
            if self.fast:
                self.keys.set("f11", True)
            factor = REFERENCE_JUMP / max(peak - GROUND, 1)
            self.gravity_cache[road] = float(np.clip((factor - 1.0) * 5.0, -1.0, 1.0))
        return self.gravity_cache[road]

    def observe_memory(self, row, x, y, speed, gravity):
        """The pilot's inputs from the game's real state (shared with the matched simulator)."""
        raw = np.frombuffer(self.ram.read(MEM_ROAD + row * ROW_BYTES, OBS_ROWS * ROW_BYTES),
                            np.uint16).reshape(OBS_ROWS, C.LANES)
        return build_obs(raw, x, y, speed, gravity)

    def last_tunnel(self, length):
        """(first row, lanes) of a tunnel in the road's last rows, or None."""
        n = min(length, 4)
        cells = np.frombuffer(self.ram.read(MEM_ROAD + (length - n) * ROW_BYTES, n * ROW_BYTES),
                              np.uint16).reshape(n, C.LANES)
        rows, lanes = np.nonzero((cells >> 8) == CELL_TUNNEL)
        if not len(rows):
            return None
        return length - n + int(rows.min()), set(int(v) for v in lanes)

    def goto(self, road, tries=3):
        """Menu, select, start. The menu is sometimes slow to settle, so retry before giving up.
        When fast-forwarding, the menus run at normal speed (fast-forwarded, the cursor races past
        before a key press registers) and the speed comes back once the race has started."""
        pace = self.pace
        if self.fast:
            self.keys.set("f11", False)
            self.pace = 1.0
        try:
            for attempt in range(tries):
                try:
                    self.to_menu()
                    self.select(road, timeout=8.0)
                    self.start_race()
                    return
                except WindowLost:
                    if attempt == tries - 1 or not self.alive():
                        self.save_screen(f"goto_road{road}")
                        raise
                    time.sleep(1.0)
        finally:
            if self.fast:
                self.pace = pace
                self.keys.set("f11", True)

    def save_screen(self, tag):
        """Keep a picture of an unexpected screen for diagnosis (at most 20)."""
        folder = os.path.join(HERE, "..", "runs", "logs", "screens")
        try:
            os.makedirs(folder, exist_ok=True)
            if len(os.listdir(folder)) < 20:
                cv2.imwrite(os.path.join(folder, f"{tag}_{self.proc.pid}_{int(time.time())}.png"), self.grabber.grab())
        except Exception:
            pass

    def fly(self, net, road, hz=30.0, max_seconds=150.0, carrier=None, handover_row=None, jump_mode="tap"):
        """One attempt, using the game's own state. Returns dict(progress, won, why, ...).

        With a carrier, that pilot flies the road and `net` takes over at `handover_row` (default: 12
        rows from the end), so candidates practise a hard stretch without first surviving everything
        before it.
        """
        if not (self.road == road and self.restarting and self.await_restart()):
            self.goto(road)
        self.road, self.restarting = road, False
        length = self.road_length()
        start_row = self.read_ship()[0]
        gravity = self.measure_gravity(road)
        handover = length - HANDOVER_ROWS if handover_row is None else min(handover_row, length - HANDOVER_ROWS)
        handover = max(start_row, handover)
        handed = carrier is None
        handed_ticks = 0
        progress, prev_y = 0.0, GROUND
        approach_ticks, approach_good, jump_ticks = 0, 0, 0
        tunnel_end = self.last_tunnel(length)
        t0 = last_move = last_check = time.time()
        last_pos, moved, why, ticks, tunnels, was_tunnel = -1.0, False, "timeout", 0, 0, False
        reached = False
        steering = Steering()
        tap, last_y = JumpTap(), GROUND
        cell = 0
        state = self.read_ship()
        try:
            while True:
                # decide once per game update: wait until the game moves the ship on (or briefly,
                # when it's standing still). An independent clock made identical pilots fly differently.
                wait_from = time.perf_counter()
                while True:
                    now_state = self.read_ship()
                    if now_state != state or time.perf_counter() - wait_from > 0.045:
                        break
                    time.sleep(0.001)
                state = now_state
                loop = time.time()
                ticks += 1
                row, frac, x, y, speed = state
                pos = row + frac
                if row < start_row - 1 or row > MAX_ROWS - OBS_ROWS - 1:
                    why = "screen changed"             # the road restarted under us
                    break
                if pos > last_pos + 1e-4:
                    last_pos, last_move = pos, loop
                    moved = moved or pos > start_row + 0.05
                lane = int(np.clip(lane_of(x), 0, C.LANES - 1))
                cell = struct.unpack("<H", self.ram.read(MEM_ROAD + row * ROW_BYTES + lane * 2, 2))[0]
                in_tunnel = cell >> 8 == CELL_TUNNEL
                tunnels += in_tunnel and not was_tunnel and y <= GROUND + 400    # through, not over
                was_tunnel = in_tunnel
                if y == 0:
                    # entering the last tunnel snaps height straight to 0; a fall sinks there gradually
                    why = "finished" if pos >= length - 1.5 and prev_y >= GROUND - 1000 else "fell"
                    break
                prev_y = y
                if cell >> 8 == 0 and cell & 255 == TILE_DEADLY and y <= GROUND + 400:
                    why = "burned"
                    break
                if tunnel_end and tunnel_end[0] - APPROACH_ROWS <= pos < tunnel_end[0]:
                    # the run-in to the last tunnel: on the ground, in the tunnel's lane?
                    approach_ticks += 1
                    approach_good += y <= GROUND + 400 and lane in tunnel_end[1]
                if not handed and pos >= handover:
                    handed = True
                # only rows covered at road level or above count: off an edge the ship sinks slowly
                # while still moving forward, and those rows must not be credited
                level = y >= GROUND - BELOW_ROAD
                if handed and level:
                    origin = handover if carrier is not None else start_row
                    progress = max(progress, min(0.99, (pos - origin) / max(1, length - origin)))   # 100%: finish only
                reached = reached or (level and pos >= length - 1.5)
                frozen = loop - last_move
                if moved and speed == 0 and frozen > 0.3:
                    why = "crashed"
                    break
                if frozen > (1.5 if moved else 5.0):
                    why = "stalled"
                    break
                if loop - t0 > max_seconds:
                    break
                if loop - last_check > 0.25:           # the screen is still the backstop
                    last_check = loop
                    dark, lit = read_bar(self.grab())
                    if dark + lit <= 300:
                        why = "screen changed"
                        break
                obs = self.observe_memory(row, x, y, speed, gravity)
                out = (net if handed else carrier).activate(obs)
                steer = steering(out[0])
                thr = 1 if out[1] > 0.33 else (-1 if out[1] < -0.33 else 0)
                if jump_mode == "hold":                           # the key follows the output
                    jump = out[2] > 0.5
                else:                                             # a press per jump, not a held key
                    jump = tap(out[2], y > last_y and y > GROUND)
                last_y = y
                if handed:
                    jump_ticks += jump
                    handed_ticks += 1
                self.keys.set("left", steer < 0)
                self.keys.set("right", steer > 0)
                self.keys.set("up", thr > 0)
                self.keys.set("down", thr < 0)
                self.keys.set("space", jump)
        finally:
            for n in ("left", "right", "up", "down", "space"):
                self.keys.set(n, False)
        won = why == "finished"
        end_score = approach_good / approach_ticks if approach_ticks else 0.0
        if won:
            progress = 1.0
            end_score = 1.0
            time.sleep(5.5)                            # "Road Completed", then the road menu
        self.restarting = why in ("fell", "crashed", "burned", "screen changed")
        self.faded = why == "screen changed"
        secs = time.time() - t0
        return {"progress": progress, "won": won, "seconds": secs, "why": why,
                "hz": ticks / max(secs, 1e-3), "tunnels": int(tunnels), "cell": cell, "length": length,
                "reached_end": reached, "end_score": end_score, "jump_frac": jump_ticks / max(handed_ticks, 1),
                "skipped": not handed}

    def close(self):
        try:
            self.proc.terminate()
        except OSError:
            pass
