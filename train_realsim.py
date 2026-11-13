"""Evolve pilots in the simulator matched to the real game, starting from the real-game population.

Pilots see the same inputs here as on the real game, so the result can go straight back to it:
    python train_realsim.py --out runs/realsim
    python train_real.py --play runs/realsim/best.pkl --roads 0 1 2      # try it on the real game

The roads are the simulator's own procedural layouts; difficulty rises as pilots start winning.
"""
import argparse
import csv
import glob
import os
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import neat
import numpy as np

from starlane import migrate, realroads, realsim
from starlane.neat_evo import config_from_text

_CFG = {}


def _fly_road(args):
    genomes, cfg_text, seed, difficulty, vary, hard = args
    if cfg_text not in _CFG:
        _CFG.clear()
        _CFG[cfg_text] = config_from_text(cfg_text)
    cfg = _CFG[cfg_text]
    nets = [neat.nn.FeedForwardNetwork.create(g, cfg) for g in genomes]
    road = realroads.gauntlet(seed) if hard else realroads.generate(seed, difficulty)
    return realsim.fly(nets, road, vary)


def score(r):
    return 1000 * r["progress"] + 500 * r["end_score"] + 1000 * r["won"]


def latest_checkpoint(folder):
    cks = glob.glob(os.path.join(folder, "neat-checkpoint-*"))
    return max(cks, key=lambda s: int(s.rsplit("-", 1)[1])) if cks else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gens", type=int, default=1000)
    p.add_argument("--pop", type=int, default=150)
    p.add_argument("--roads", type=int, default=12, help="simulated roads per generation")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--difficulty", type=float, default=0.0)
    p.add_argument("--promote", type=float, default=0.5, help="champion win rate that raises difficulty")
    p.add_argument("--target-species", type=int, default=10)
    p.add_argument("--gauntlets", type=int, default=4,
                   help="extra full-difficulty courses per generation, packed with every kind of challenge")
    p.add_argument("--vary", type=float, default=1.0,
                   help="nudge the physics per road (0 = exactly as measured): about +-8%% acceleration and "
                        "steering, +-3%% jump, +-4%% gravity, and one road in five reacts a game update later")
    p.add_argument("--seed-from", default="runs/real", help="real-game run whose population to start from")
    p.add_argument("--out", default="runs/realsim")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--link-real", default="runs/real", help="real-game run to swap pilots with ('' to switch off)")
    p.add_argument("--imports", type=int, default=10, help="real pilots brought in every 10 generations")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    state = {"gen": 0, "difficulty": a.difficulty, "ema": 0.0, "best": -1.0}
    ck = latest_checkpoint(a.out) if a.resume else None
    if ck:
        pop = neat.Checkpointer.restore_checkpoint(ck)
        cfg_text = open(os.path.join(a.out, "neat.cfg")).read()
        state = pickle.load(open(os.path.join(a.out, "state.pkl"), "rb"))
        print(f"resumed from {ck}, generation {state['gen']}, difficulty {state['difficulty']:.2f}")
    else:
        src = latest_checkpoint(a.seed_from)
        pop = neat.Checkpointer.restore_checkpoint(src)
        cfg_text = open(os.path.join(a.seed_from, "neat.cfg")).read()
        pop.config.pop_size = a.pop
        pop.config.species_set_config.compatibility_threshold = 3.0
        pop.species = pop.config.species_set_type(pop.config.species_set_config, pop.reporters)
        pop.species.speciate(pop.config, pop.population, pop.generation)
        with open(os.path.join(a.out, "neat.cfg"), "w") as f:
            f.write(cfg_text)
        print(f"seeded from {src}: {len(pop.population)} pilots, growing to {a.pop}")
    pop.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(a.out, "neat-checkpoint-")))

    log_path = os.path.join(a.out, "log.csv")
    new_log = not (ck and os.path.exists(log_path))
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["gen", "difficulty", "best", "mean", "champ_wins", "pop_win_rate", "species", "secs"])
    pool = ProcessPoolExecutor(a.workers)

    link = {"prev_keys": None, "seen": None}

    def eval_genomes(genomes, cfg):
        t0 = time.time()
        imported = []
        if a.link_real and state["gen"] % 10 == 0:
            incoming = migrate.load(os.path.join(a.link_real, "migrants.pkl"))
            if incoming and link["prev_keys"] is not None and incoming["gen"] != link["seen"]:
                imported = migrate.bring_in(genomes, link["prev_keys"], incoming["genomes"], cfg.genome_config,
                                            pop.reproduction.innovation_tracker, a.imports)
                link["seen"] = incoming["gen"]
        ids, gs = zip(*genomes)
        seeds = [7_000_000 + state["gen"] * 97 + i for i in range(a.roads)]
        hard_seeds = [9_000_000 + state["gen"] * 97 + i for i in range(a.gauntlets)]
        jobs = ([(list(gs), cfg_text, s, state["difficulty"], a.vary, False) for s in seeds]
                + [(list(gs), cfg_text, s, 1.0, a.vary, True) for s in hard_seeds])
        flown = list(pool.map(_fly_road, jobs))
        per_road, per_hard = flown[:len(seeds)], flown[len(seeds):]
        fits = np.zeros(len(gs))
        wins = np.zeros(len(gs))
        for res in per_road:
            wins += [r["won"] for r in res]
        for res in flown:                              # gauntlets count in fitness like any road
            fits += [score(r) for r in res]
        fits /= len(flown)
        hard_prog = np.array([[r["progress"] for r in res] for res in per_hard]) if per_hard else None
        for g, f in zip(gs, fits):
            g.fitness = float(f)
        b = int(np.argmax(fits))
        link["prev_keys"] = set(ids)
        order = np.argsort(-fits)
        migrate.export(os.path.join(a.out, "migrants.pkl"), [gs[i] for i in order[:5]], state["gen"], "sim")
        if imported:
            ranks = [int(np.nonzero(order == ids.index(g))[0][0]) + 1 for g in imported]
            print(f"       <- {len(imported)} real-game pilots brought in; best placed {min(ranks)} of {len(gs)}", flush=True)
        champ_rate = wins[b] / len(per_road)
        state["ema"] = 0.7 * state["ema"] + 0.3 * champ_rate
        ends = Counter(r["why"] for res in per_road for r in res)
        n_species = len(pop.species.species)
        secs = time.time() - t0
        print(f"gen {state['gen']:4d}  diff {state['difficulty']:.2f}  best {fits[b]:7.1f}  mean {fits.mean():7.1f}  "
              f"champ wins {int(wins[b])}/{len(per_road)} (avg {state['ema']:.0%})  population wins {wins.mean() / len(per_road):.0%}  "
              f"species {n_species} (thr {cfg.species_set_config.compatibility_threshold:.1f})  "
              + (f"gauntlets: champ {hard_prog[:, b].mean():.0%}, furthest {hard_prog.max():.0%}, "
                 f"wins {int(sum(r['won'] for res in per_hard for r in res))}  " if hard_prog is not None else "")
              + f"[{', '.join(f'{k} {v}' for k, v in ends.most_common())}]  {secs:.1f}s", flush=True)
        w.writerow([state["gen"], f"{state['difficulty']:.2f}", f"{fits[b]:.1f}", f"{fits.mean():.1f}",
                    int(wins[b]), f"{wins.mean() / len(per_road):.3f}", n_species, f"{secs:.1f}"])
        log.flush()
        blob = {"kind": "neat", "genome": gs[b], "config_text": cfg_text, "obs_mode": "vision",
                "meta": {"gen": state["gen"], "difficulty": state["difficulty"], "fitness": float(fits[b]),
                         "sim": True}}
        pickle.dump(blob, open(os.path.join(a.out, "latest_best.pkl"), "wb"))
        rank = state["difficulty"] * 10_000 + fits[b]
        if rank > state["best"]:
            state["best"] = rank
            pickle.dump(blob, open(os.path.join(a.out, "best.pkl"), "wb"))
        if state["ema"] >= a.promote and state["difficulty"] < 1.0:
            state["difficulty"] = round(min(1.0, state["difficulty"] + 0.05), 2)
            state["ema"] = 0.5 * a.promote
            print(f"       -> difficulty raised to {state['difficulty']:.2f}", flush=True)
        sc = cfg.species_set_config
        # small fixed steps within limits: species counts react with a delay, so big steps overshoot
        if n_species > a.target_species:
            sc.compatibility_threshold = min(8.0, sc.compatibility_threshold + 0.1)
        elif n_species < a.target_species:
            sc.compatibility_threshold = max(0.5, sc.compatibility_threshold - 0.1)
        state["gen"] += 1
        pickle.dump(state, open(os.path.join(a.out, "state.pkl"), "wb"))

    try:
        pop.run(eval_genomes, a.gens)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        pool.shutdown()
        log.close()


if __name__ == "__main__":
    main()
