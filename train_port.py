"""Train pilots in the skyroads-sdl port's game logic (headless, hundreds of times faster than DOSBox).

    python train_port.py --resume --out runs/port                 # continue
    python train_port.py --seed-from runs/real --out runs/port    # start from the DOSBox run's pilots

Each generation, every pilot:
  * flies each current road twice, with DOSBox-like frame timing jitter (a different pattern each
    time, so pilots can't rely on exact timing);
  * practises from several points along each road: the pilot that got furthest on a road flies it,
    and the candidate takes over at evenly spaced rows up to a few rows before where that pilot
    stopped (or near the end, once a road can be finished).
Fitness is each pilot's standing on every road and practice point (train_real.rank_fitness), so every
road counts the same. Training starts on Road 1 alone; when the best pilot has finished every current
road on every flight for --unlock-after generations running, the next road is added. Roads already
mastered earn a bonus (--keep-weight) for the share of their flights that finish, so they stay learned.
Pilots use the same inputs as in DOSBox, so they can be checked there (check_transfer.py).
"""
import argparse
import copy
import csv
import os
import pickle
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import neat
import numpy as np

from starlane.neat_evo import config_from_text
from train_real import flight_score, latest_checkpoint, rank_fitness

ROADS = 30
END_KEY = 1000                  # result keys from here on are end-of-road practice (ranked like roads)
RUNUP_KEY = -5000               # practice starts carried by a scripted run-up (ranked as practice)

# Scripted run-ups to practice starts beyond spots no pilot has passed yet, per road (0-based), as
# (practice start row, RunUp plan). Road 3: the left lane ends at row 49 and the road goes on only via
# short middle lanes (rows 46-49) and a strip on the far right. No pilot landed on the middle lanes in
# 166 generations, yet a scripted sweep (jump about row 43 steering right, then jump again steering to
# the far right) finished Road 3 in 4,212 of 10,512 attempts. These starts let pilots learn the second
# jump and the ending; the first jump they still have to find on full flights.
# The timings depend on how the evolved pilot arrives at row 40, so each road's run-up keeps the carrier
# it was swept with (state["runup_carriers"], pinned the first time it's needed) rather than following
# the best-ever carrier, which changes.
RUNUPS = {
    2: [(47, {"take": 40, "j1": 44.0, "s1": 43.5, "t1": 3.5, "thr": 0, "j2": 48.0}),    # on the middle lanes
        (50, {"take": 40, "j1": 44.0, "s1": 43.5, "t1": 3.5, "thr": 0, "j2": 46.5})],   # on the right strip
}
_W = {}

# Mutation test (gen 1241, 4 pilots finishing Roads 1 and 2, 60 offspring each): with the neat.cfg
# settings only 8% of mutated copies still finished both roads (2% after crossover with a species-mate),
# so finishers survived as elites but never multiplied. With these settings 96% did.
GENTLE = {"weight_mutate_rate": 0.1, "weight_mutate_power": 0.1, "weight_replace_rate": 0.0,
          "bias_mutate_rate": 0.1, "bias_mutate_power": 0.1, "bias_replace_rate": 0.0,
          "conn_add_prob": 0.05, "conn_delete_prob": 0.02, "node_add_prob": 0.02, "node_delete_prob": 0.0,
          "enabled_mutate_rate": 0.0}


def single_parent_breeding(share):
    """Make `share` of offspring copies of their fitter parent (then mutated) instead of crossovers.
    Crossover with a species-mate kept the Road 2 skill in only 29% of offspring even with gentle
    mutation. neat-python always crosses two parents, so this wraps the genome's crossover."""
    import random
    cls = neat.DefaultGenome
    if getattr(cls, "_orig_crossover", None) is None:
        cls._orig_crossover = cls.configure_crossover

    def crossover(self, g1, g2, config):
        if random.random() < share and g1.fitness is not None and g2.fitness is not None:
            fitter = g1 if g1.fitness >= g2.fitness else g2
            return cls._orig_crossover(self, fitter, fitter, config)
        return cls._orig_crossover(self, g1, g2, config)
    cls.configure_crossover = crossover


def explorer_mutation(share, strong):
    """Mutate `share` of offspring with the strong settings (`strong`: values of the GENTLE keys).
    Gentle mutation alone made the population near-clones (2 species) that all hit the same block on
    Road 3 (gens 1266-1421 stuck at 12%); a few strongly mutated offspring explore new behaviour while
    the gentle majority keeps passing on the roads already learned."""
    import random
    cls = neat.DefaultGenome
    if getattr(cls, "_orig_mutate", None) is None:
        cls._orig_mutate = cls.mutate

    def mutate(self, config):
        if random.random() >= share:
            return cls._orig_mutate(self, config)
        saved = {k: getattr(config, k) for k in strong}
        for k, v in strong.items():
            setattr(config, k, v)
        try:
            return cls._orig_mutate(self, config)
        finally:
            for k, v in saved.items():
                setattr(config, k, v)
    cls.mutate = mutate


def below_normal():
    """Training yields the CPU to anything interactive, e.g. the DOSBox window of check_transfer.py."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def _work(args):
    """Fly a batch of pilots on a list of jobs. Runs in a worker process with its own game logic."""
    pilots, cfg_text, jobs, carriers, play = args          # play: {"jump", "frame_ticks", "jitter"}
    if "env" not in _W:
        below_normal()
        from starlane.portenv import PortEnv
        _W["env"] = PortEnv()
    if _W.get("cfg_text") != cfg_text:
        _W["cfg_text"], _W["cfg"] = cfg_text, config_from_text(cfg_text)
    env, cfg = _W["env"], _W["cfg"]
    carrier_nets = {road: neat.nn.FeedForwardNetwork.create(g, cfg) for road, g in carriers.items()}
    out = {}
    for gid, genome in pilots:
        net = neat.nn.FeedForwardNetwork.create(genome, cfg)
        res = {}
        for key, road, handover, seed, *plan in jobs:
            carrier = carrier_nets.get(road) if handover is not None else None
            if handover is not None and carrier is None:
                continue
            if plan and plan[0]:
                from starlane.portenv import RunUp
                g = plan[0]["carrier"]
                if _W.get("runup_id") != (id(g), g.key):
                    _W["runup_id"], _W["runup_net"] = (id(g), g.key), neat.nn.FeedForwardNetwork.create(g, cfg)
                carrier = RunUp(_W["runup_net"], plan[0])  # stateful: a fresh one for every flight
            res[key] = env.fly(net, road, carrier=carrier, handover_row=handover,
                               seed=seed if play["jitter"] else 0, frame_ticks=play["frame_ticks"],
                               jump=play["jump"])
        out[gid] = res
    return out


INPUTS = 222                    # starlane/realobs.build_obs


def widen_inputs(pop, cfg_text, path):
    """Give an existing population the inputs added since it was made (163 -> 165: the road's end;
    165 -> 180: the jump-aware clear path; 180 -> 195: the same with cell types read correctly;
    195 -> 222: special tiles ahead, what the ship stands on, fuel and oxygen),
    each connected to every output at zero strength so evolution can start tuning them at once."""
    gc = pop.config.genome_config
    have = gc.num_inputs
    if have >= INPUTS:
        return cfg_text
    old = f"num_inputs            = {have}\n"
    assert old in cfg_text, "neat.cfg layout changed"
    cfg_text = cfg_text.replace(old, f"num_inputs            = {INPUTS}\n")
    gc.num_inputs = INPUTS
    gc.input_keys = [-i - 1 for i in range(INPUTS)]
    gc.innovation_tracker = pop.reproduction.innovation_tracker
    for g in pop.population.values():
        for k in range(have + 1, INPUTS + 1):
            for out in gc.output_keys:
                if (-k, out) not in g.connections:
                    g.add_connection(gc, -k, out, 0.0, True)
    with open(path, "w") as f:
        f.write(cfg_text)
    print(f"inputs widened from {have} to {INPUTS}", flush=True)
    return cfg_text


def merge(flights):
    """Several flights of one pilot on one road -> one result with the average score."""
    n = len(flights)
    return {"progress": sum(f["progress"] for f in flights) / n, "end_score": sum(f["end_score"] for f in flights) / n,
            "won": sum(f["won"] for f in flights) / n, "skipped": all(f["skipped"] for f in flights)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/port")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed-from", default="runs/real",
                   help="NEAT run whose latest population to start from (with --fresh: whose settings to use)")
    p.add_argument("--fresh", action="store_true", help="start from new random pilots")
    p.add_argument("--pop", type=int, default=200)
    p.add_argument("--gens", type=int, default=100000)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--flights", type=int, default=1, help="flights per road per pilot (more only helps with --jitter)")
    p.add_argument("--jump", choices=["hold", "tap"], default="hold",
                   help="hold: the jump key follows the output, as in the game; tap: one jump per switch-on")
    p.add_argument("--frame-ticks", type=float, default=1.0,
                   help="game ticks per decision: 1 = every update (36/s); 1.385 = DOSBox's ~26 frames/s")
    p.add_argument("--jitter", action="store_true",
                   help="vary each frame between 1 and 2 ticks as DOSBox does (needs several --flights)")
    p.add_argument("--practice-points", type=int, default=4, help="takeover points per road")
    p.add_argument("--no-end-practice", dest="end_practice", action="store_false",
                   help="don't add the end-of-road takeovers (12 and 6 rows before the end)")
    p.add_argument("--roads", type=int, nargs="+", default=[0], help="starting roads (0 = Road 1)")
    p.add_argument("--unlock-after", type=int, default=5,
                   help="generations in a row the top pilots must finish every current road before the next is added")
    p.add_argument("--keep-weight", type=float, default=600.0,
                   help="fitness bonus for each already-mastered road, times the share of its flights finished")
    p.add_argument("--win-weight", type=float, default=400.0,
                   help="fitness bonus for finishing the newest road (ranks alone barely separate a "
                        "finish from falling off at 99%%)")
    p.add_argument("--mutation", choices=["gentle", "cfg"], default="gentle",
                   help="gentle: small weight changes (GENTLE); cfg: the stronger neat.cfg settings")
    p.add_argument("--single-parent", type=float, default=0.75,
                   help="share of offspring bred from one parent instead of crossover")
    p.add_argument("--no-runups", dest="runups", action="store_false",
                   help="skip the scripted run-up practice starts (RUNUPS)")
    p.add_argument("--explorers", type=float, default=0.1,
                   help="with --mutation gentle: share of offspring mutated with the strong neat.cfg settings")
    p.add_argument("--seeds", type=int, default=7,
                   help="protected seeds: the furthest stepping-stone niche in each lane of the newest road, "
                        "kept even if they fail older roads")
    p.add_argument("--seed-children", type=int, default=25,
                   help="gently mutated children of the seeds added each generation")
    p.add_argument("--min-threshold", type=float, default=0.05,
                   help="lowest species compatibility threshold (gentle mutation makes pilots very similar)")
    p.add_argument("--unlock-share", type=float, default=0.5,
                   help="share of the top 20 pilots that must finish every current road on every flight")
    p.add_argument("--target-species", type=int, default=12)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    below_normal()

    state = {"gen": 0, "best": -1.0, "roads": list(a.roads), "carriers": {}, "reach": {}}
    ck = latest_checkpoint(a.out) if a.resume else None
    if ck:
        pop = neat.Checkpointer.restore_checkpoint(ck)
        cfg_text = open(os.path.join(a.out, "neat.cfg")).read()
        state = pickle.load(open(os.path.join(a.out, "state.pkl"), "rb"))
        print(f"resumed from {ck}: generation {state['gen']}, roads {[r + 1 for r in state['roads']]}", flush=True)
    elif a.fresh:
        # brand-new random pilots, with the same network layout and NEAT settings as before
        cfg_text = open(os.path.join(a.seed_from, "neat.cfg")).read()
        with open(os.path.join(a.out, "neat.cfg"), "w") as f:
            f.write(cfg_text)
        config = config_from_text(cfg_text)
        config.pop_size = a.pop
        pop = neat.Population(config)
        print(f"fresh start: {len(pop.population)} new pilots", flush=True)
    else:
        src = latest_checkpoint(a.seed_from)
        pop = neat.Checkpointer.restore_checkpoint(src)
        cfg_text = open(os.path.join(a.seed_from, "neat.cfg")).read()
        with open(os.path.join(a.out, "neat.cfg"), "w") as f:
            f.write(cfg_text)
        for sp in pop.species.species.values():      # a new scale of fitness: start species histories afresh
            sp.fitness_history, sp.fitness, sp.last_improved = [], None, pop.generation
        print(f"seeded from {src}: {len(pop.population)} pilots, growing to {a.pop}", flush=True)
    cfg_text = widen_inputs(pop, cfg_text, os.path.join(a.out, "neat.cfg"))
    pop.config.pop_size = a.pop
    pop.config.reproduction_config.min_species_size = 1
    if a.mutation == "gentle":
        # the checkpoint's config may already hold the gentle values: take the strong ones from neat.cfg
        cfg_gc = config_from_text(cfg_text).genome_config
        strong = {k: getattr(cfg_gc, k) for k in GENTLE}
        for k, v in GENTLE.items():
            setattr(pop.config.genome_config, k, v)
        explorer_mutation(a.explorers, strong)
    single_parent_breeding(a.single_parent)
    print(f"mutation {a.mutation} (explorers {a.explorers:.0%}), single-parent offspring "
          f"{a.single_parent:.0%}, species threshold floor {a.min_threshold}", flush=True)
    pop.add_reporter(neat.Checkpointer(10, filename_prefix=os.path.join(a.out, "neat-checkpoint-")))

    log_path = os.path.join(a.out, "log.csv")
    new_log = not (ck and os.path.exists(log_path))
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["gen", "roads", "best_points", "best_detail", "furthest", "wins", "species", "secs"])
    pool = ProcessPoolExecutor(a.workers)

    def practice_rows(road, length):
        """Takeover rows along the road, up to a few rows before where the carrier stopped."""
        reach = state["reach"].get(road)
        if reach is None:
            return []
        top = length - 12 if reach >= length - 1.5 else int(reach) - 6
        if top < 4:
            return []
        rows = {int(r) for r in np.linspace(4, top, a.practice_points)}
        if reach < length - 1.5 and int(reach) - 3 > top:
            rows.add(int(reach) - 3)          # right up against the spot where everyone gets stuck
            # and with room to set up for it: Road 3's row-52 wall needs a crossing started by row ~44,
            # which the start at reach - 3 (row 49) left no room for
            if int(reach) - 9 > 4:
                rows.add(int(reach) - 9)
        return sorted(rows)

    lengths = {}

    # Protected seeds: the pilots that got furthest on the newest road, kept even if they fail older
    # roads. Without them, the one explorer that got past Road 3's first block (27%) ranked near the
    # bottom (it had lost Roads 1-2) and died out at once. Each generation a few new offspring are
    # replaced by unchanged seeds and gently mutated seed children, so a lucky one-off becomes a line
    # that can win the older roads back. The archive keeps the best a.seeds by (newest-road progress,
    # then points on the older roads), so a child that goes as far and flies the old roads better
    # replaces its parent: a small hill-climb towards pilots that do both.
    def frontier_key(r, roads):
        # whole rows reached on the newest road: pilots stopping at the same spot tie and are ranked by
        # the older roads (comparing progress to 4 decimals let a seed reaching 86.49% with 407 points
        # on Roads 1-2 outrank one reaching 86.38% with 1700, though both fell into the same gap)
        new = roads[-1]
        row = int(3 + float(r[new]["progress"]) * (lengths[new] - 3)) + (1000 if r[new]["won"] else 0)
        old = [flight_score(r[rd]) for rd in roads[:-1]]
        return (row, round(sum(old) / len(old), 1) if old else 0.0)

    def plant_seeds(genomes, cfg):
        seeds = state.get("frontier", [])
        if not seeds and a.seeds and state["roads"][-1] in state["carriers"]:
            seeds = state["frontier"] = [((0.0, 0.0), copy.deepcopy(state["carriers"][state["roads"][-1]]))]
        if not seeds:
            return 0
        fresh = [g for _, g in genomes if g.fitness is None]           # not this generation's elites
        random.shuffle(fresh)
        gc = cfg.genome_config
        saved = {k: getattr(gc, k) for k in GENTLE}
        for k, v in GENTLE.items():
            setattr(gc, k, v)
        try:
            n = 0
            for i, g in enumerate(fresh[:len(seeds) + a.seed_children]):
                # children lean towards the furthest seed (about 45% of them for 5 seeds)
                src = seeds[i][1] if i < len(seeds) else seeds[int(random.random() ** 2 * len(seeds))][1]
                g.nodes, g.connections = copy.deepcopy(src.nodes), copy.deepcopy(src.connections)
                if i >= len(seeds):
                    (getattr(neat.DefaultGenome, "_orig_mutate", None) or neat.DefaultGenome.mutate)(g, gc)
                n += 1
        finally:
            for k, v in saved.items():
                setattr(gc, k, v)
        return n

    # Stepping-stone niches (MAP-Elites): every spot a pilot last stood on before its newest-road flight
    # ended, (row, lane), keeps the pilot that got there and flies the older roads best. Ranking seeds by
    # distance alone stalled at Road 3 row 52 for 240 generations: getting on takes two jumps (left lane
    # -> middle lanes -> right-hand strip), and a pilot that made the first jump and then fell scored no
    # better than one riding off the end, so the half-way solution was always thrown away. The seeds
    # planted each generation are the furthest niche in each lane, so a pilot that reached the middle
    # lanes is kept and its children only need to learn the second jump.
    def update_seeds(genomes, flown, res, roads):
        if not a.seeds:
            return
        new = roads[-1]
        niches = state.setdefault("niches", {})
        for gid, g in genomes:
            d = flown[gid][(new, 0)]
            row, lane = (1000, 0) if d["won"] else d["rest"]
            old = frontier_key(res[gid], roads)[1]
            if (row, lane) not in niches or old > niches[(row, lane)][0]:
                niches[(row, lane)] = (old, copy.deepcopy(g))
        per_lane = {}
        for (row, lane), (old, g) in niches.items():
            if lane not in per_lane or (row, old) > per_lane[lane][:2]:
                per_lane[lane] = (row, old, g)
        for key in [k for k in niches if k[0] < per_lane[k[1]][0] - 10]:
            del niches[key]                         # far behind its lane's best: keeps state.pkl small
        top = sorted((((row, old, lane), g) for lane, (row, old, g) in per_lane.items()),
                     key=lambda kg: kg[0], reverse=True)[:a.seeds]
        state["frontier"] = top

    def eval_genomes(genomes, cfg):
        t0 = time.time()
        gen = state["gen"]
        roads = state["roads"]
        if not lengths:
            from starlane.portenv import PortEnv
            env = PortEnv()
            for r in range(ROADS):
                lengths[r] = env.load(r)
        jobs = []
        for road in roads:
            for i in range(a.flights):
                jobs.append(((road, i), road, None, 1 + gen * 1009 + road * 97 + i))
            for j, row in enumerate(practice_rows(road, lengths[road])):
                jobs.append((-(1 + road * 100 + j), road, row, 7 + gen * 1013 + road * 89 + j))
            # the ending: once a pilot can carry others to a road's end, everyone also starts 12 and
            # 6 rows before it. These count like whole roads (keys >= END_KEY rank as roads), so
            # entering the final tunnel matters as much as getting through a road.
            if a.runups and road == roads[-1] and road in RUNUPS:
                pinned = state.setdefault("runup_carriers", {})
                if road not in pinned and road in state["carriers"]:
                    pinned[road] = copy.deepcopy(state["carriers"][road])
                if road in pinned:
                    for j, (row, plan) in enumerate(RUNUPS[road]):
                        jobs.append((RUNUP_KEY - road * 10 - j, road, row, 17 + gen * 1021 + road * 79 + j,
                                     dict(plan, carrier=pinned[road])))
            reach = state["reach"].get(road)
            if a.end_practice and reach is not None and reach >= lengths[road] - 1.5:
                for j, back in enumerate((12, 6)):
                    jobs.append((END_KEY + road * 10 + j, road, lengths[road] - back, 13 + gen * 1019 + road * 83 + j))
        planted = plant_seeds(genomes, cfg)
        carriers = {r: g for r, g in state["carriers"].items() if r in roads}
        items = list(genomes)
        chunk = max(1, len(items) // (a.workers * 3))
        play = {"jump": a.jump, "frame_ticks": a.frame_ticks, "jitter": a.jitter}
        batches = [(items[i:i + chunk], cfg_text, jobs, carriers, play) for i in range(0, len(items), chunk)]
        flown = {}
        for part in pool.map(_work, batches):
            flown.update(part)

        res = {}
        for gid, d in flown.items():
            r = {}
            for road in roads:
                r[road] = merge([d[(road, i)] for i in range(a.flights)])
            for k, v in d.items():
                if not isinstance(k, tuple):
                    r[k] = v
            res[gid] = r
        fits = rank_fitness(res)
        update_seeds(genomes, flown, res, roads)
        # mastered roads (every road but the newest) should stay finished: each pilot gets a bonus for
        # the share of its flights there that finish. Without it, gains on the new road came at the
        # cost of the old ones (Road 1 finishes fell from ~400 to ~10 per 38 generations). With a bonus
        # that outranked everything (5000), pilots that cleared Road 2's first gap but missed one Road 1
        # finish were always dropped; at --keep-weight 600, losing Road 1 entirely still costs more than
        # anything the new road can give, but one missed finish (150) can be outweighed by real progress.
        # The bonus is per mastered road, not an average over them: averaged, losing one of two old roads
        # cost only 300, less than the 400 for finishing the newest road, and Road 2 finishers fell from
        # ~150 to 9 per generation while pilots learned Road 3 (gens 2569-2682).
        mastered = roads[:-1]
        if mastered:
            for gid in fits:
                kept = sum(res[gid][rd]["won"] for rd in mastered)
                fits[gid] += a.keep_weight * kept
        # the newest road: finishing it must clearly beat getting 99% of the way. With ranks alone,
        # Road 2 finishers (1-13 of 500 per generation, gens 774-1409) never took over and died out.
        newest = roads[-1]
        for gid in fits:
            fits[gid] += a.win_weight * res[gid][newest]["won"]
        for gid, g in genomes:
            g.fitness = fits[gid]
        best_id = max(fits, key=fits.get)
        top = sorted(fits, key=fits.get, reverse=True)[:20]
        steady = sum(all(res[gid][rd]["won"] == 1 for rd in roads) for gid in top) / len(top)
        points = {gid: sum(flight_score(res[gid][rd]) for rd in roads) / len(roads) for gid in res}

        # carriers for next generation: the pilot that got furthest on each road (best of its flights),
        # kept until someone gets further. Taking only this generation's furthest let the practice at
        # Road 2's end vanish when its finishers died out, which made the end harder to relearn.
        for road in roads:
            far = {gid: max(d[(road, i)]["progress"] for i in range(a.flights)) for gid, d in flown.items()}
            top = max(far, key=far.get)
            reach = 3 + far[top] * (lengths[road] - 3)
            if reach >= state["reach"].get(road, -1.0) or road not in state["carriers"]:
                state["carriers"][road] = dict(genomes)[top]
                state["reach"][road] = reach

        wins = Counter()
        for d in flown.values():
            for k, v in d.items():
                if v["won"]:
                    if isinstance(k, tuple):
                        wins["r" + str(k[0] + 1)] += 1
                    elif k >= END_KEY:
                        wins["end" + str((k - END_KEY) // 10 + 1)] += 1
                    else:
                        wins["practice"] += 1
        why = Counter(v["why"] for d in flown.values() for k, v in d.items() if isinstance(k, tuple))
        champ = res[best_id]
        detail = " ".join(f"R{rd + 1}:{champ[rd]['progress']:.0%}" for rd in roads)
        furthest = " ".join(f"R{rd + 1}:{(state['reach'][rd] - 3) / (lengths[rd] - 3):.0%}" for rd in roads)
        n_species = len(pop.species.species) if pop.species.species else 0
        secs = time.time() - t0
        print(f"gen {gen:5d}  roads {len(roads)}  best [{detail}] ({points[best_id]:.0f} pts)  furthest [{furthest}]  "
              f"steady {steady:.0%}  "
              f"wins {dict(wins)}  species {n_species} (thr {cfg.species_set_config.compatibility_threshold:.2f})  [{', '.join(f'{k} {v}' for k, v in why.most_common())}]  "
              f"{len(items)} pilots x {len(jobs)} flights  {secs:.1f}s", flush=True)
        if planted:
            seeds_txt = "  ".join(("won" if k[0] >= 1000 else f"row {k[0]} lane {k[2]}") + f"/old {k[1]:.0f}"
                                  for k, _ in state.get("frontier", []))
            seeds_txt = f"{len(state.get('niches', {}))} niches on R{roads[-1] + 1}; seeds {seeds_txt}"
            print(f"       seeds: {planted} planted; archive {seeds_txt}", flush=True)
        w.writerow([gen, " ".join(str(r + 1) for r in roads), f"{points[best_id]:.1f}", detail, furthest,
                    sum(wins.values()), n_species, f"{secs:.1f}"])
        log.flush()

        champ_g = dict(genomes)[best_id]
        blob = {"kind": "neat", "genome": champ_g, "config_text": cfg_text, "obs_mode": "vision",
                "meta": {"gen": gen, "roads": list(roads), "fitness": fits[best_id], "port": True}}
        with open(os.path.join(a.out, "latest_best.pkl"), "wb") as f:
            pickle.dump(blob, f)
        if points[best_id] > state["best"]:
            state["best"] = points[best_id]
            with open(os.path.join(a.out, "best.pkl"), "wb") as f:
                pickle.dump(blob, f)

        # one road at a time: the next road is added once at least a.unlock_share of the top 20 pilots
        # finish every current road on every flight, in a.unlock_after generations running (one lucky
        # champion isn't enough: that skill was lost again as soon as the next road came in)
        if steady >= a.unlock_share:
            state["streak"] = state.get("streak", 0) + 1
        else:
            state["streak"] = 0
        if state["streak"] >= a.unlock_after and max(roads) < ROADS - 1:
            roads.append(max(roads) + 1)
            state["best"], state["streak"], state["frontier"], state["niches"] = -1.0, 0, [], {}
            print(f"       -> road {roads[-1] + 1} added", flush=True)

        # steer the species count towards the target: a step in proportion to how far off it is
        # (fixed small steps were far too slow: the count drifted to 50 species of 3-4 pilots)
        sc = cfg.species_set_config
        if n_species:
            # NEAT never merges species (they only die out), so the count reacts slowly to the threshold:
            # small steps and a moderate cap, or the threshold runs far ahead and, once old species die
            # out, hardly any new ones can form
            step = float(np.clip(0.02 * (n_species - a.target_species), -0.2, 0.2))
            sc.compatibility_threshold = float(np.clip(sc.compatibility_threshold + step, a.min_threshold, 15.0))
        # neat-python never merges species: they only die out after max_stagnation generations. When the
        # explorers arrived, each formed its own species (2 -> 105) and the count stayed put while the
        # threshold ran up towards its cap. Far above target, sort everyone into species afresh.
        if n_species > 2 * a.target_species:
            ss = pop.species
            ss.species, ss.genome_to_species = {}, {}
            ss.speciate(cfg, dict(genomes), pop.generation)
            print(f"       -> {n_species} species re-sorted into {len(ss.species)} "
                  f"(thr {sc.compatibility_threshold:.2f})", flush=True)
        state["gen"] += 1
        with open(os.path.join(a.out, "state.pkl"), "wb") as f:
            pickle.dump(state, f)

    try:
        pop.run(eval_genomes, a.gens)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
