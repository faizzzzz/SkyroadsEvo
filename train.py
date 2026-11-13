"""Evolve Starlane pilots.

    python train.py --gens 300 --workers 8
    python train.py --obs vision --noise 0.05 --out runs/vision   # for the real-game bridge
"""
import argparse
import csv
import os
import time

import numpy as np

from starlane import config as C
from starlane.evolve import ES, GA, Evaluator
from starlane.policy import MLPSpec, save_genome
from starlane.sim import obs_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=300)
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--levels", type=int, default=16, help="levels per generation")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--obs", choices=["full", "vision"], default="full")
    p.add_argument("--noise", type=float, default=0.0, help="gaussian noise on observations")
    p.add_argument("--hidden", type=int, nargs="+", default=[32, 16])
    p.add_argument("--difficulty", type=float, default=0.0, help="starting difficulty")
    p.add_argument("--promote", type=float, default=0.6,
                   help="raise difficulty when the elite win rate reaches this")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="runs/default")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--algo", choices=["es", "ga"], default="ga",
                   help="es: OpenAI-style evolution strategy; ga: genetic algorithm")
    p.add_argument("--sigma", type=float, default=None, help="ES noise std / GA initial sigma")
    p.add_argument("--lr", type=float, default=0.03, help="ES learning rate")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    spec = MLPSpec(obs_size(a.obs), tuple(a.hidden))
    if a.algo == "es":
        algo = ES(spec, pop_size=a.pop, sigma=a.sigma or 0.05, lr=a.lr, seed=a.seed)
    else:
        algo = GA(spec, pop_size=a.pop, sigma0=a.sigma or 0.08, seed=a.seed)
    diff, start_gen = a.difficulty, 0
    ckpt = os.path.join(a.out, "population.npz")
    if a.resume and os.path.exists(ckpt):
        f = np.load(ckpt)
        algo.load(f)
        diff, start_gen = float(f["difficulty"]), int(f["gen"]) + 1
        print(f"resumed at gen {start_gen}, difficulty {diff:.2f}")

    print(f"genome size {spec.dim}, population {a.pop}, {a.levels} levels/gen, "
          f"{a.workers} workers, obs={a.obs}, algo={a.algo}")
    ev = Evaluator(a.workers)
    log_path = os.path.join(a.out, "log.csv")
    new_log = not (a.resume and os.path.exists(log_path))
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["gen", "difficulty", "best", "mean", "champ_win", "pop_win", "sigma", "secs"])
    best_ever = -1.0
    win_ema = 0.0   # smoothed win rate, so one lucky generation doesn't promote
    try:
        for gen in range(start_gen, start_gen + a.gens):
            t0 = time.time()
            seeds = [a.seed * 1_000_003 + gen * 101 + i for i in range(a.levels)]
            pop = algo.ask()
            fit, wins, reasons = ev.evaluate(pop, spec, a.obs, seeds, diff, a.noise)
            champ, best, elite_win = algo.champion(fit, wins)
            win_ema = 0.7 * win_ema + 0.3 * elite_win
            # most common ways the population fails
            flat = reasons.ravel()
            counts = np.bincount(flat, minlength=len(C.REASONS))
            top = ", ".join(f"{C.REASONS[i]} {counts[i] / flat.size:.0%}"
                            for i in np.argsort(-counts)[:3] if counts[i])
            secs = time.time() - t0
            print(f"gen {gen:4d}  diff {diff:.2f}  best {fit[best]:7.1f}  mean {fit.mean():7.1f}  "
                  f"champ win {elite_win:4.0%} (avg {win_ema:4.0%})  sigma {algo.step_size:.3f}  [{top}]  {secs:.1f}s",
                  flush=True)
            w.writerow([gen, f"{diff:.3f}", f"{fit[best]:.1f}", f"{fit.mean():.1f}",
                        f"{elite_win:.3f}", f"{wins.mean():.3f}", f"{algo.step_size:.4f}",
                        f"{secs:.1f}"])
            log.flush()

            meta = {"gen": gen, "difficulty": diff, "fitness": float(fit[best]),
                    "win_rate": float(wins[best])}
            save_genome(os.path.join(a.out, "latest_best.npz"), champ, spec, a.obs, meta)
            # "best.npz" favours the hardest difficulty reached, then fitness
            score = diff * 10_000 + fit[best]
            if score > best_ever:
                best_ever = score
                save_genome(os.path.join(a.out, "best.npz"), champ, spec, a.obs, meta)

            if win_ema >= a.promote and diff < 1.0:
                diff = min(1.0, diff + 0.05)
                win_ema = 0.5 * a.promote
                print(f"           -> difficulty raised to {diff:.2f}")

            algo.tell(fit)
            np.savez(ckpt, difficulty=diff, gen=gen, **algo.state())
    except KeyboardInterrupt:
        print("stopped")
    finally:
        ev.close()
        log.close()


if __name__ == "__main__":
    main()
