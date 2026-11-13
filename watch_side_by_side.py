"""Watch the best pilot fly live in DOSBox and in the port, side by side on the second monitor.

    python watch_side_by_side.py                         # runs/port/latest_best.pkl on its training roads
    python watch_side_by_side.py runs/port/best.pkl --roads 1 --rounds 1

The DOSBox window runs at normal speed. Next to it a window shows the port, drawn by its own
renderer and paced to real game speed. Each road starts in both at the same moment (when the DOSBox
pilot takes control). Every round reloads the pilot file, so you see the latest best. Close the port
window to stop.
"""
import argparse
import multiprocessing as mp
import os
import pickle
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "Port (skyroads-sdl)"


def port_side(conn, x, y, jump, frame_ticks):
    """Child process: fly the port live in an OpenCV window whenever told to."""
    import cv2
    import neat
    import numpy as np
    from starlane.neat_evo import config_from_text
    from starlane.portenv import PortEnv

    env = PortEnv(dll=os.path.join(HERE, "ext", "skyenv", "skyenv_draw.dll"))
    rgb = np.zeros((200, 320, 3), np.uint8)
    cv2.namedWindow(TITLE, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(TITLE, x, y)
    last = np.zeros((480, 640, 3), np.uint8)

    def show(img, caption):
        frame = img.copy()
        cv2.rectangle(frame, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(frame, caption, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(TITLE, frame)
        cv2.waitKey(1)
        return cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) >= 1

    show(last, "waiting for DOSBox...")
    while True:
        if not conn.poll(0.03):
            cv2.waitKey(1)
            if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                conn.send(("closed",))
                break
            continue
        msg = conn.recv()
        if msg is None:
            break
        road, genome, cfg_text, label = msg
        net = neat.nn.FeedForwardNetwork.create(genome, config_from_text(cfg_text))
        caption = f"{label}  road {road + 1}"
        t0 = time.perf_counter()
        is_open = [True]

        def on_frame(s):
            nonlocal last
            env.lib.env_frame_rgb(rgb.ctypes.data)
            last = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (640, 480), interpolation=cv2.INTER_NEAREST)
            wait = t0 + s.tick / 36.0 - time.perf_counter()      # real game speed: 36 ticks a second
            if wait > 0:
                time.sleep(wait)
            is_open[0] = show(last, caption) and is_open[0]

        r = env.fly(net, road, on_frame=on_frame, jump=jump, frame_ticks=frame_ticks)
        is_open[0] = show(last, f"{caption}:  {r['progress']:.0%} - {r['why']}") and is_open[0]
        conn.send(("done", r["progress"], r["why"], is_open[0]))
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pilot", nargs="?", default="runs/port/latest_best.pkl")
    p.add_argument("--roads", type=int, nargs="+", default=None,
                   help="0 = Road 1; default: the roads currently in training (runs/port/state.pkl)")
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--jump", choices=["hold", "tap"], default="hold", help="as the pilot was trained")
    p.add_argument("--frame-ticks", type=float, default=1.0,
                   help="port: game ticks per decision, as trained (DOSBox always decides per drawn frame)")
    a = p.parse_args()

    import neat
    from bridge import live
    from starlane.neat_evo import config_from_text

    # DOSBox in the middle of three slots across the second monitor, the port window right of it
    # (the hourly check's DOSBox window uses the first slot)
    win = live.GameWindow(1, live.RoadVision(live.Palettes()), total=3, fast=False)
    rect = live.wintypes.RECT()
    live.user32.GetWindowRect(win.hwnd, live.ctypes.byref(rect))
    here, there = mp.Pipe()
    side = mp.Process(target=port_side, args=(there, rect.right + 4, rect.top, a.jump, a.frame_ticks), daemon=True)
    side.start()
    try:
        time.sleep(1.5)
        for _ in range(a.rounds):
            blob = pickle.load(open(os.path.join(HERE, a.pilot), "rb"))
            net = neat.nn.FeedForwardNetwork.create(blob["genome"], config_from_text(blob["config_text"]))
            label = f"{os.path.basename(a.pilot)} gen {blob['meta']['gen']}"
            roads = a.roads or blob["meta"].get("roads") or [0]
            for road in roads:
                started = [False]

                class Synced:
                    def activate(self, obs):
                        if not started[0]:           # the DOSBox pilot takes control: start the port too
                            started[0] = True
                            here.send((road, blob["genome"], blob["config_text"], label))
                        return net.activate(obs)

                if here.poll() and here.recv()[0] == "closed":
                    return
                rd = win.fly(Synced(), road, max_seconds=90, jump_mode=a.jump)
                msg = here.recv() if started[0] else ("done", 0.0, "-", True)
                if msg[0] == "closed" or not msg[3]:
                    return
                print(f"{label}  road {road + 1}:  DOSBox {rd['progress']:.0%} {rd['why']}   "
                      f"port {msg[1]:.0%} {msg[2]}", flush=True)
                time.sleep(2.0)
    finally:
        try:
            here.send(None)
        except (OSError, BrokenPipeError):
            pass
        win.keys.release_all()
        win.close()
        side.join(timeout=3)


if __name__ == "__main__":
    main()
