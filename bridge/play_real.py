"""Let an evolved pilot drive a real game window through screen capture + key presses.

    python -m bridge.play_real runs/vision/best.npz --window SKYROADS.EXE

Only pilots trained with --obs vision work here. Capture and keys go straight to
the game window, so it can sit behind other windows. Hold F12 to stop at once.
A debug window shows the sampled road cells: red = hole, green = floor,
yellow = raised. Use it to check the calibration lines up with the road.

Use --dry-run to watch what the pilot *would* press without sending any keys.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from starlane import config as C
from starlane.pilot import Pilot
from starlane.policy import to_actions
from bridge.vision import Vision
from bridge.winapi import VK_F12, PostKeys, WindowGrabber, find_window, key_down_now

HERE = os.path.dirname(__file__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("genome")
    p.add_argument("--window", default="SKYROADS.EXE")
    p.add_argument("--calib", default=os.path.join(HERE, "calibration.json"))
    p.add_argument("--hz", type=float, default=30.0, help="decision rate")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-debug", action="store_true")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until F12/Esc)")
    p.add_argument("--save-frames", help="folder for an annotated frame every half second")
    a = p.parse_args()

    pilot = Pilot(a.genome)
    if pilot.obs_mode != "vision":
        raise SystemExit("this genome was trained with --obs full; retrain with --obs vision")
    hwnd, title = find_window(a.window)
    if not hwnd:
        raise SystemExit(f"no window with '{a.window}' in its title")
    with open(a.calib) as f:
        cal = json.load(f)
    km = cal["keys"]
    vision = Vision(a.calib)
    grab = WindowGrabber(hwnd)     # works while other windows cover the game
    keys = PostKeys(hwnd, [km["left"], km["right"], km["accel"], km["brake"], km["jump"]])
    speed, since_jump = 0.0, 1.0
    dt_target = 1.0 / a.hz
    print(f"driving '{title}'. Hold F12 to stop. You can keep using other windows.")
    started, last_save, saved = time.perf_counter(), 0.0, 0
    lives, was_dark, life_start, last_refresh = 0, False, started, started
    if a.save_frames:
        os.makedirs(a.save_frames, exist_ok=True)

    try:
        last = time.perf_counter()
        while True:
            if key_down_now(VK_F12):
                print("F12 - stopping")
                break
            img = grab.grab()
            dark = img[:600].mean() < 60        # the screen fades out when a road restarts
            if dark and not was_dark:
                lives += 1
                print(f"ship lost, life {lives} over at {time.perf_counter() - life_start:.1f}s", flush=True)
                speed = 0.0
            if was_dark and not dark:
                life_start = time.perf_counter()
                keys.refresh()
            was_dark = dark
            obs, cells, pix, x = vision.observe(img, speed, since_jump)
            out = pilot.act(obs)
            steer, thr, jump = (int(v[0]) for v in to_actions(out))

            now = time.perf_counter()
            dt, last = now - last, now
            # dead-reckon speed from our own throttle (the screen does not tell us)
            acc = C.ACCEL if thr > 0 else (-C.BRAKE if thr < 0 else 0.0)
            speed = float(np.clip(speed + acc * dt * cal.get("speed_scale", 1.0), 0, C.V_MAX))
            since_jump = 0.0 if jump else since_jump + dt

            live = not a.dry_run
            if live and now - last_refresh > 0.5:
                keys.refresh()
                last_refresh = now
            if live:
                keys.set(km["left"], steer < 0)
                keys.set(km["right"], steer > 0)
                keys.set(km["accel"], thr > 0)
                keys.set(km["brake"], thr < 0)
                keys.set(km["jump"], bool(jump))
            else:
                keys.release_all()

            elapsed = time.perf_counter() - started
            if a.seconds and elapsed > a.seconds:
                break
            due = a.save_frames and elapsed - last_save >= 0.5
            if not a.no_debug or due:
                dbg = img.copy()
                colors = {-1: (0, 0, 255), 0: (0, 200, 0), 1: (0, 220, 255)}
                for (ix, iy), v in zip(pix.reshape(-1, 2), cells.ravel()):
                    cv2.circle(dbg, (int(ix), int(iy)), 2, colors[int(v)], -1)
                txt = (f"x={x:.2f} v~{speed:4.1f} steer={steer:+d} thr={thr:+d} jump={int(jump)} "
                       f"{'LIVE' if live else 'dry run'}")
                cv2.putText(dbg, txt, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                if due:
                    cv2.imwrite(os.path.join(a.save_frames, f"t{saved:03d}.png"), dbg)
                    print(f"t={elapsed:5.1f}s {txt}", flush=True)
                    saved, last_save = saved + 1, elapsed
                if not a.no_debug:
                    s = max(1, int(800 / dbg.shape[1]))
                    cv2.imshow("pilot", cv2.resize(dbg, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST))
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
            spare = dt_target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
    finally:
        keys.release_all()
        if not a.no_debug:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
