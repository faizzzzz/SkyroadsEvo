"""Record GIFs of pilots from different points in training, flown in the skyroads-sdl port.

    python record_progress.py              # writes docs/media/*.gif

Each clip picks a pilot from a checkpoint (or a saved pilot file), flies it on one road with the port's
own renderer, and saves every third frame (12 fps) at the game's 320x200 with a caption. Clips are
listed in CLIPS.
"""
import glob
import os
import pickle
import re
import sys

import cv2
import neat
import numpy as np

from starlane.neat_evo import config_from_text
from starlane.portenv import PortEnv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "docs", "media")
CFG = config_from_text(open(os.path.join(HERE, "runs", "port", "neat.cfg")).read())
FLY = dict(jump="hold", frame_ticks=1.0)

# (file name, caption, source, road (0-based), how to pick the pilot)
#   source: a checkpoint generation number in runs/port, or a pilot file
#   pick: "furthest" (most progress), "finisher" (finishes this road, preferring pilots that also
#         finish the earlier roads), "best" (the pilot as saved)
CLIPS = [
    ("01_road1_gen20.gif", "Gen 20: reaches the end, hops over the tunnel", 20, 0, "furthest"),
    ("02_road1_gen100.gif", "Gen 100: first Road 1 finishes", 100, 0, "finisher"),
    ("03_road2_gen800.gif", "Gen 800: Road 2 finished", 800, 1, "finisher"),
    ("04_road3_wall.gif", "Road 3: all 500 pilots hit this block", 1300, 2, "furthest"),
    ("05_road3_finish.gif", "Gen 3050: Road 3 finished", "latest", 2, "finisher"),
    ("06_road4.gif", "Road 4: latest best pilot", os.path.join("runs", "port", "latest_best.pkl"), 3, "best"),
]


def checkpoint(gen):
    files = glob.glob(os.path.join(HERE, "runs", "port", "neat-checkpoint-*"))
    nums = {int(re.search(r"(\d+)$", f).group(1)): f for f in files}
    if gen == "latest":
        return nums[max(nums)]
    return nums[min(nums, key=lambda n: abs(n - gen))]


def pick(source, road, how, env):
    if isinstance(source, str) and source.endswith(".pkl"):
        return pickle.load(open(os.path.join(HERE, source), "rb"))["genome"], "saved pilot"
    path = checkpoint(source)
    genomes = list(neat.Checkpointer.restore_checkpoint(path).population.values())
    scored = []
    for g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, CFG)
        r = env.fly(net, road, **FLY)
        earlier = sum(env.fly(net, rd, **FLY)["won"] for rd in range(road)) if how == "finisher" and r["won"] else 0
        scored.append(((r["won"], earlier, r["progress"]), g))
    key, g = max(scored, key=lambda kg: kg[0])
    return g, f"{os.path.basename(path)}, {key}"


def record(genome, road, caption, path):
    env = PortEnv(dll=os.path.join(HERE, "ext", "skyenv", "skyenv_draw.dll"))
    rgb = np.zeros((200, 320, 3), np.uint8)
    frames = []

    def on_frame(s):
        if s.frame % 3 == 0:
            env.lib.env_frame_rgb(rgb.ctypes.data)
            img = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)[:122])   # road only, no dashboard
            cv2.rectangle(img, (0, 0), (320, 13), (0, 0, 0), -1)
            cv2.putText(img, caption, (4, 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(img)

    r = env.fly(neat.nn.FeedForwardNetwork.create(genome, CFG), road, on_frame=on_frame, **FLY)
    for _ in range(18):                      # hold the last frame for 1.5 s
        frames.append(frames[-1])
    anim = cv2.Animation()
    anim.loop_count = 0
    anim.durations = [83] * len(frames)
    anim.frames = frames
    cv2.imwriteanimation(path, anim)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    only = set(sys.argv[1:])
    env = PortEnv()
    for name, caption, source, road, how in CLIPS:
        if only and name not in only:
            continue
        genome, note = pick(source, road, how, env)
        r = record(genome, road, caption, os.path.join(OUT, name))
        size = os.path.getsize(os.path.join(OUT, name)) / 1e6
        print(f"{name}: {note} -> {r['progress']:.0%} {r['why']}  ({size:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
