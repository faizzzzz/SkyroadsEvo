"""Evolve Starlane pilots with NEAT (the algorithm MarIO uses).

    python train_neat.py --out runs/neat_full
    python train_neat.py --obs vision --noise 0.05 --out runs/neat_vision

Ctrl+C stops cleanly; --resume continues from the latest checkpoint.
"""
import argparse
import csv
import glob
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import neat
import numpy as np

from starlane import config as C
from starlane.neat_evo import config_from_text, evaluate_neat, make_config_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=500)
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--levels", type=int, default=16, help="levels per generation")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--obs", choices=["full", "vision"], default="full")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--promote", type=float, default=0.6)
    p.add_argument("--conn-frac", type=float, default=0.03,
                   help="fraction of input->output links each starting genome gets")
    p.add_argument("--compat", type=float, default=3.0,
                   help="starting species compatibility threshold (lower = more species)")
    p.add_argument("--target-species", type=int, default=12,
                   help="nudge the threshold each generation toward this many species (0 = fixed)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="runs/neat")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cfg_path = os.path.join(a.out, "neat.cfg")
    if a.resume and os.path.exists(cfg_path):
        cfg_text = open(cfg_path).read()
    else:
        cfg_text = make_config_text(a.obs, a.pop, a.conn_frac, a.compat)
        with open(cfg_path, "w") as f:
            f.write(cfg_text)
    config = config_from_text(cfg_text)

    state = {"diff": a.difficulty, "gen": 0, "win_ema": 0.0, "best_ever": -1.0}
    ckpts = sorted(glob.glob(os.path.join(a.out, "neat-checkpoint-*")),
                   key=lambda s: int(s.rsplit("-", 1)[1]))
    if a.resume and ckpts:
        pop = neat.Checkpointer.restore_checkpoint(ckpts[-1])
        with open(os.path.join(a.out, "curriculum.pkl"), "rb") as f:
            state = pickle.load(f)
        if "compat" in state:
            pop.config.species_set_config.compatibility_threshold = state["compat"]
        print(f"resumed from {ckpts[-1]}, difficulty {state['diff']:.2f}")
    else:
        pop = neat.Population(config, seed=a.seed)
    pop.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(a.out, "neat-checkpoint-")))

    log_path = os.path.join(a.out, "log.csv")
    new_log = not (a.resume and os.path.exists(log_path))
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["gen", "difficulty", "best", "mean", "champ_win", "pop_win",
                    "species", "champ_nodes", "champ_conns", "secs"])
    pool = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
    print(f"NEAT: {config.genome_config.num_inputs} inputs, population {a.pop}, "
          f"{a.levels} levels/gen, {a.workers} workers, obs={a.obs}")

    def eval_genomes(genomes, cfg):
        t0 = time.time()
        gen = state["gen"]
        ids, gs = zip(*genomes)
        seeds = [a.seed * 1_000_003 + gen * 101 + i for i in range(a.levels)]
        fit, wins, reasons = evaluate_neat(pool, list(gs), cfg_text, a.obs, seeds, state["diff"], a.noise)
        for g, f in zip(gs, fit):
            g.fitness = float(f)
        b = int(np.argmax(fit))
        champ = gs[b]
        state["win_ema"] = 0.7 * state["win_ema"] + 0.3 * float(wins[b])
        flat = reasons.ravel()
        counts = np.bincount(flat, minlength=len(C.REASONS))
        top = ", ".join(f"{C.REASONS[i]} {counts[i] / flat.size:.0%}"
                        for i in np.argsort(-counts)[:3] if counts[i])
        n_nodes = len(champ.nodes)
        n_conns = sum(1 for c in champ.connections.values() if c.enabled)
        n_species = len(pop.species.species) if pop.species.species else 0
        secs = time.time() - t0
        print(f"gen {gen:4d}  diff {state['diff']:.2f}  best {fit[b]:7.1f}  mean {fit.mean():7.1f}  "
              f"champ win {wins[b]:4.0%} (avg {state['win_ema']:4.0%})  species {n_species:2d} "
              f"(thr {cfg.species_set_config.compatibility_threshold:.1f})  "
              f"champ {n_nodes}n/{n_conns}c  [{top}]  {secs:.1f}s", flush=True)
        w.writerow([gen, f"{state['diff']:.3f}", f"{fit[b]:.1f}", f"{fit.mean():.1f}",
                    f"{wins[b]:.3f}", f"{wins.mean():.3f}", n_species, n_nodes, n_conns, f"{secs:.1f}"])
        log.flush()

        blob = {"kind": "neat", "genome": champ, "config_text": cfg_text, "obs_mode": a.obs,
                "meta": {"gen": gen, "difficulty": state["diff"], "fitness": float(fit[b]),
                         "win_rate": float(wins[b])}}
        with open(os.path.join(a.out, "latest_best.pkl"), "wb") as f:
            pickle.dump(blob, f)
        score = state["diff"] * 10_000 + fit[b]
        if score > state["best_ever"]:
            state["best_ever"] = score
            with open(os.path.join(a.out, "best.pkl"), "wb") as f:
                pickle.dump(blob, f)

        if state["win_ema"] >= a.promote and state["diff"] < 1.0:
            state["diff"] = min(1.0, state["diff"] + 0.05)
            state["win_ema"] = 0.5 * a.promote
            print(f"           -> difficulty raised to {state['diff']:.2f}")
        # keep the number of species near the target by nudging the threshold
        sc = cfg.species_set_config
        if a.target_species and n_species:
            if n_species > a.target_species:
                sc.compatibility_threshold += 0.1
            elif n_species < a.target_species:
                sc.compatibility_threshold = max(0.3, sc.compatibility_threshold - 0.1)
        state["compat"] = sc.compatibility_threshold
        state["gen"] = gen + 1
        with open(os.path.join(a.out, "curriculum.pkl"), "wb") as f:
            pickle.dump(state, f)

    try:
        pop.run(eval_genomes, a.gens)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if pool:
            pool.shutdown()
        log.close()


if __name__ == "__main__":
    main()
