"""Does training carry over to the real game? Every so often, fly trained pilots in DOSBox.

    python check_transfer.py                 # check now, then every hour
    python check_transfer.py --once          # a single check

By default each check flies the port run's latest and best-ever pilots (runs/port) on roads 1-3 in
one DOSBox window at normal speed, and next to each the port's own prediction for the same pilot,
appending the results to runs/logs/transfer.txt. --sim / --real add pilots from those runs.
Progress counts only rows covered at road level.
"""
import argparse
import os
import pickle
import time

import neat

from bridge import live
from starlane import migrate
from starlane.neat_evo import config_from_text

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(results):
    return ",".join(f"{r['progress']:.0%}{'W' if r['won'] else ''}" for r in results)


def check(a, cfg):
    pilots = []
    head = time.strftime("%Y-%m-%d %H:%M")
    if a.port:
        for name in ("latest_best", "best"):
            path = os.path.join(HERE, a.port, name + ".pkl")
            if os.path.exists(path):
                blob = pickle.load(open(path, "rb"))
                pilots.append((f"port {name.replace('_', ' ')} (gen {blob['meta']['gen']})", blob["genome"]))
    if a.sim:
        sim = migrate.load(os.path.join(HERE, a.sim, "migrants.pkl"))
        sim_state = pickle.load(open(os.path.join(HERE, a.sim, "state.pkl"), "rb"))
        pilots += [(f"sim #{i + 1}", g) for i, g in enumerate(sim["genomes"][:a.pilots])] if sim else []
        head += f"  sim gen {sim_state['gen']} difficulty {sim_state['difficulty']:.2f}"
    if a.real:
        real_best = pickle.load(open(os.path.join(HERE, a.real, "latest_best.pkl"), "rb"))
        pilots.append((f"dosbox-trained (gen {real_best['meta']['gen']})", real_best["genome"]))
    port_env = None
    if a.port:
        from starlane.portenv import PortEnv
        port_env = PortEnv()
    lines = []
    box = {"win": None}

    def opened():
        if box["win"] is None:
            box["win"] = live.GameWindow(0, live.RoadVision(live.Palettes()), fast=False)   # normal speed
            time.sleep(1.5)
        return box["win"]

    def fly(net, road):
        # on a busy PC the menu sometimes misses a key press; the training windows recover by
        # reopening, and so does this
        for attempt in range(3):
            try:
                return opened().fly(net, road, max_seconds=90)
            except (live.WindowLost, OSError):
                if attempt == 2:
                    raise
                box["win"].keys.release_all()
                box["win"].close()
                box["win"] = None

    try:
        for name, genome in pilots:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            per_road = {road: [fly(net, road) for _ in range(a.flights)] for road in a.roads}
            mean = sum(r["progress"] for rr in per_road.values() for r in rr) / sum(map(len, per_road.values()))
            lines.append(f"    {name:28s} DOSBox mean {mean:4.0%}   "
                         + "   ".join(f"road {road + 1}: {fmt(rr)}" for road, rr in per_road.items()))
            if port_env is not None:     # what the port predicts for the same pilot (4 timing patterns)
                pred = {road: [port_env.fly(net, road, seed=s) for s in (11, 22, 33, 44)] for road in a.roads}
                pmean = sum(r["progress"] for rr in pred.values() for r in rr) / sum(map(len, pred.values()))
                lines.append(f"    {'':28s} port   mean {pmean:4.0%}   "
                             + "   ".join(f"road {road + 1}: {fmt(rr)}" for road, rr in pred.items()))
    finally:
        if box["win"] is not None:
            box["win"].keys.release_all()
            box["win"].close()
    text = head + "\n" + "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "runs", "logs", "transfer.txt"), "a") as f:
        f.write(text)
    print(text, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="runs/port", help="port-trained run whose pilots to check ('' to skip)")
    p.add_argument("--sim", default="", help="generated-road sim run to check too, e.g. runs/realsim")
    p.add_argument("--real", default="", help="DOSBox-trained run to check too, e.g. runs/real")
    p.add_argument("--pilots", type=int, default=3, help="sim pilots per check")
    p.add_argument("--flights", type=int, default=2, help="flights per pilot per road")
    p.add_argument("--roads", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--every", type=float, default=60.0, help="minutes between checks")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    cfg = config_from_text(open(os.path.join(HERE, a.port or a.real or a.sim, "neat.cfg")).read())
    while True:
        t0 = time.time()
        try:
            check(a, cfg)
        except (live.WindowLost, OSError) as e:
            print(f"check failed: {e}", flush=True)
        if a.once:
            break
        time.sleep(max(60.0, a.every * 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
