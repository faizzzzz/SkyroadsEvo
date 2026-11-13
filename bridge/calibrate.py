"""Interactive calibration for the real-game bridge.

Start your own copy of the game in DOSBox, get to a point where a straight,
full-width stretch of road is on screen, then run:

    python -m bridge.calibrate --window SKYROADS.EXE

Steps (follow the prompt in the window title / console):
  1. click the 4 road corners: near-left, near-right, far-right, far-left
  2. click 1+ points of empty space / void, press ENTER
  3. click the ship, press ENTER
  4. (optional) click 1+ raised blocks, press ENTER (or just ENTER to skip)
Then type how many rows ahead of the ship the near and far edges are.
Press SPACE at any time to re-grab the screen.
"""
import argparse
import json
import os

import cv2
import numpy as np

from bridge.winapi import Grabber, find_window

HERE = os.path.dirname(__file__)
STEPS = ["click 4 road corners: near-left, near-right, far-right, far-left",
         "click empty space (void) colours, then ENTER",
         "click the ship, then ENTER",
         "click raised-block colours (optional), then ENTER"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", default="SKYROADS.EXE", help="part of the window title")
    p.add_argument("--out", default=os.path.join(HERE, "calibration.json"))
    p.add_argument("--image", help="calibrate from a saved screenshot instead of a live window")
    a = p.parse_args()

    if a.image:
        grab = lambda: cv2.imread(a.image)
    else:
        hwnd, title = find_window(a.window)
        if not hwnd:
            raise SystemExit(f"no window with '{a.window}' in its title")
        print("capturing:", title)
        g = Grabber(hwnd)
        grab = g.grab

    img = grab()
    scale = max(1, int(900 / img.shape[1]))  # enlarge small DOS screens for clicking
    step, quad, bg, ship, raised = 0, [], [], [], []

    def on_click(ev, x, y, *_):
        nonlocal step
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = x // scale, y // scale
        c = img[y, x].tolist()
        if step == 0:
            quad.append([x, y])
            if len(quad) == 4:
                step = 1
        elif step == 1:
            bg.append(c)
        elif step == 2:
            ship.append((x, y, c))
        elif step == 3:
            raised.append(c)
        print(f"step {step + 1}: ({x},{y}) colour BGR {c}")

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_click)
    while step < 4:
        view = img.copy()
        if len(quad) > 1:
            cv2.polylines(view, [np.array(quad)], len(quad) == 4, (0, 255, 255), 1)
        for x, y, _ in ship:
            cv2.circle(view, (x, y), 3, (0, 0, 255), -1)
        view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.setWindowTitle("calibrate", f"step {step + 1}: {STEPS[min(step, 3)]}")
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 and step >= 1:
            if step == 1 and not bg:
                continue
            if step == 2 and not ship:
                continue
            step += 1
        elif k == 32:
            img = grab()
        elif k == 27:
            return
    cv2.destroyAllWindows()

    near = float(input("rows ahead of the ship at the NEAR edge (e.g. 0.5): ") or 0.5)
    far = float(input("rows ahead of the ship at the FAR edge (e.g. 20): ") or 20)
    sy = int(np.mean([s[1] for s in ship]))
    cal = {
        "road_quad": quad, "near_rows": near, "far_rows": far,
        "background": bg, "raised": raised,
        "ship": {"color": np.mean([s[2] for s in ship], 0).tolist(),
                 "band": [max(0, sy - 25), min(img.shape[0], sy + 25)]},
        "bg_tol": 40, "ship_tol": 40,
        "keys": {"left": "left", "right": "right", "accel": "up", "brake": "down", "jump": "space"},
        "speed_scale": 1.0,
    }
    with open(a.out, "w") as f:
        json.dump(cal, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
