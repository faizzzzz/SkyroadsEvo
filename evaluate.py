"""Benchmark a saved pilot on fixed held-out levels at several difficulties.

    python evaluate.py runs/full/best.npz
"""
import argparse
from collections import Counter

import numpy as np

from starlane import config as C
from starlane.level import generate
from starlane.pilot import Pilot
from starlane.policy import to_actions
from starlane.sim import BatchSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("genome")
    p.add_argument("--levels", type=int, default=30)
    p.add_argument("--noise", type=float, default=0.0)
    a = p.parse_args()
    pilot = Pilot(a.genome)
    print(f"{a.genome}: {pilot.describe()}")
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        reasons, fits = Counter(), []
        for s in range(a.levels):
            sim = BatchSim(generate(900_000 + s, d), 1, pilot.obs_mode, a.noise,
                           np.random.default_rng(s))
            while sim.alive[0]:
                sim.step(*to_actions(pilot.act(sim.observe())))
            reasons[C.REASONS[sim.reason[0]]] += 1
            fits.append(sim.fitness()[0])
        wins = reasons.get("win", 0)
        print(f"difficulty {d:.2f}: won {wins:3d}/{a.levels}  mean fitness {np.mean(fits):7.1f}  "
              f"{dict(reasons.most_common())}")


if __name__ == "__main__":
    main()
