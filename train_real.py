"""Evolve pilots by letting them play the real game, on its real roads.

Starts from the best NEAT pilots evolved in the simulator, then keeps evolving them
with fitness measured in actual game windows.

    python train_real.py --windows 6 --out runs/real
    python train_real.py --resume --out runs/real
    python train_real.py --play runs/real/best.pkl --roads 0 1 2

Hold F12 or press Ctrl+C to stop. Windows go on a second monitor if there is one.
"""
import argparse
import csv
from collections import Counter
import glob
import os
import multiprocessing as mp
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.connection import wait

import neat
import numpy as np

from bridge import live
from bridge.winapi import VK_F12, key_down_now
from starlane import migrate
from starlane.neat_evo import config_from_text, evaluate_neat


def latest_checkpoint(folder):
    cks = glob.glob(os.path.join(folder, "neat-checkpoint-*"))
    return max(cks, key=lambda s: int(s.rsplit("-", 1)[1])) if cks else None


def add_gravity_input(pop, cfg_text, path):
    """Give an existing population one more input (road gravity), linked to the jump output
    with zero weight so evolution can start tuning it straight away."""
    gc = pop.config.genome_config
    if gc.num_inputs != 162:                   # only the original layout needs this
        return cfg_text
    want = gc.num_inputs + 1
    old = f"num_inputs            = {gc.num_inputs}\n"
    if old not in cfg_text:
        return cfg_text                        # already migrated
    new_text = cfg_text.replace(old, f"num_inputs            = {want}\n")
    gc.num_inputs = want
    gc.input_keys = [-i - 1 for i in range(want)]
    gc.innovation_tracker = pop.reproduction.innovation_tracker
    for g in pop.population.values():
        if (-want, 2) not in g.connections:
            g.add_connection(gc, -want, 2, 0.0, True)
    with open(path, "w") as f:
        f.write(new_text)
    print(f"added a gravity input: {want} inputs", flush=True)
    return new_text


def seed_population(a):
    """A small population: the sim run's latest pilots, ranked on simulated levels."""
    ck = latest_checkpoint(a.seed_from)
    if not ck:
        raise SystemExit(f"no checkpoint in {a.seed_from}")
    pop = neat.Checkpointer.restore_checkpoint(ck)
    with open(os.path.join(a.seed_from, "neat.cfg")) as f:
        cfg_text = f.read()
    ids, genomes = zip(*pop.population.items())
    with ProcessPoolExecutor(8) as ex:
        fit, _, _ = evaluate_neat(ex, list(genomes), cfg_text, "vision",
                                  [770_000 + i for i in range(12)], 0.25, 0.05)
    top = np.argsort(-fit)[:a.pop]
    print(f"seeded from {ck}: top sim fitness {fit[top[0]]:.0f}, cutoff {fit[top[-1]]:.0f}")
    pop.population = {ids[i]: genomes[i] for i in top}
    pop.config.pop_size = a.pop
    pop.species = pop.config.species_set_type(pop.config.species_set_config, pop.reporters)
    pop.species.speciate(pop.config, pop.population, pop.generation)
    return pop, cfg_text


class LocalPalettes:
    """A worker's view of the floor colours: given with each job, sightings sent back."""

    def __init__(self):
        self.known = {}
        self.seen = {}

    def floors(self, road):
        return self.known.get(road, np.zeros((0, 3), np.int16))

    def see(self, road, colour):
        key = (road, tuple(int(v) for v in colour))
        self.seen[key] = self.seen.get(key, 0) + 1

    def take(self):
        seen, self.seen = self.seen, {}
        return seen


def worker(index, total, cfg_text, conn, fast=False):
    """Owns one game window, flying whatever genome it is sent."""
    pal = LocalPalettes()
    vision = live.RoadVision(pal)
    cfg = config_from_text(cfg_text)
    try:
        win = live.GameWindow(index, vision, total, fast=fast)
    except live.WindowLost as e:
        conn.send(("dead", str(e)))
        return
    conn.send(("ready", win.proc.pid))
    while True:
        job = conn.recv()
        if job is None:
            break
        genome, road, floors, carrier, handover = job
        pal.known[road] = floors
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            if carrier is None:
                result = win.fly(net, road)
            else:                                  # the carrier may crash first; give it three tries
                carrier_net = neat.nn.FeedForwardNetwork.create(carrier, cfg)
                for _ in range(3):
                    result = win.fly(net, road, carrier=carrier_net, handover_row=handover)
                    if not result["skipped"]:
                        break
            conn.send(("done", result, pal.take()))
        except (live.WindowLost, OSError) as e:       # OSError: the game's memory couldn't be read
            win.close()
            try:
                win = live.GameWindow(index, vision, total, fast=fast)
                conn.send(("failed", str(e) or type(e).__name__, win.proc.pid))
            except (live.WindowLost, OSError) as e2:
                conn.send(("dead", str(e2)))
                return
    win.close()


class Fleet:
    """Game windows, each in its own process, flying (genome, road) jobs."""

    def __init__(self, n, palettes, cfg_text, fast=False):
        self.palettes = palettes
        self.procs, self.conns, self.pids = [], [], {}
        for i in range(n):
            here, there = mp.Pipe()
            proc = mp.Process(target=worker, args=(i, n, cfg_text, there, fast), daemon=True)
            proc.start()
            self.procs.append(proc)
            self.conns.append(here)
        self.road = {i: None for i in range(n)}     # road each window is on
        self.carriers = {}                          # real road -> pilot that carries practice flights
        self.handover = {}                          # real road -> row where the candidate takes over
        self.job = {i: None for i in range(n)}      # what each window is flying
        self.idle, self.dead = set(), set()
        pending = set(range(n))
        while pending:
            for c in wait([self.conns[i] for i in pending], timeout=60):
                i = self.conns.index(c)
                msg = c.recv()
                pending.discard(i)
                if msg[0] == "ready":
                    self.pids[i] = msg[1]
                    self.idle.add(i)
                else:
                    print(f"  window {i} failed to start: {msg[1]}", flush=True)
                    self.dead.add(i)
        if not self.idle:
            raise SystemExit("no game windows could start")

    def _take(self, pending, road, home=None):
        """Next job: the road this window is on, else its home road, else whatever has most left.
        Switching roads goes through the menu, which costs far more than the game's own restart."""
        if not pending.get(road):
            road = home if pending.get(home) else max(pending, key=lambda r: len(pending[r]))
        if not pending[road]:
            return None
        gid, genome = pending[road].pop()
        return gid, genome, road

    def evaluate(self, genomes, roads, practice=()):
        """Fly every genome on every road. Practice flights on a road are keyed -(road + 1).
        Returns {genome id: {key: result}}."""
        results = {}
        keys = list(roads) + [-(r + 1) for r in practice if r in self.carriers and r in self.handover]
        pending = {k: list(genomes) for k in keys}
        left = len(genomes) * len(keys)
        while left > 0:
            if key_down_now(VK_F12):
                raise KeyboardInterrupt
            homes = list(pending)
            for i in list(self.idle):
                job = self._take(pending, self.road[i], homes[i % len(homes)])
                if job is None:
                    break
                gid, genome, road = job
                self.job[i] = job
                self.idle.discard(i)
                real = road if road >= 0 else -road - 1
                self.conns[i].send((genome, real, self.palettes.floors(real),
                                    self.carriers[real] if road < 0 else None,
                                    self.handover.get(real) if road < 0 else None))
            busy = [self.conns[i] for i in self.job if self.job[i] is not None]
            if not busy:
                raise SystemExit("all game windows were lost")
            for c in wait(busy, timeout=0.2):
                i = self.conns.index(c)
                try:
                    msg = c.recv()
                except EOFError:
                    msg = ("dead", "process ended")
                gid, genome, road = self.job[i]
                self.job[i] = None
                if msg[0] == "done":
                    results.setdefault(gid, {})[road] = msg[1]
                    for (rd, colour), n in msg[2].items():
                        self.palettes.add(rd, colour, n)
                    self.road[i] = road
                    self.idle.add(i)
                    left -= 1
                else:
                    pending[road].append((gid, genome))
                    self.road[i] = None
                    print(f"  window {i}: {msg[1]}", flush=True)
                    if msg[0] == "failed":
                        self.pids[i] = msg[2]
                        self.idle.add(i)
                    else:
                        self.dead.add(i)
        return results

    def close(self):
        for i, c in enumerate(self.conns):
            if i not in self.dead:
                try:
                    c.send(None)
                except (OSError, BrokenPipeError):
                    pass
        for p in self.procs:
            p.join(timeout=4)
            if p.is_alive():
                p.terminate()
        for pid in self.pids.values():      # no game window left behind
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)


def flight_score(r):
    # up to 500 for driving the run-in to a road's last tunnel on the ground (the real finish needs
    # that). Jumping anywhere else is the pilot's own business.
    return 1000 * r["progress"] + 500 * r["end_score"] + 1000 * r["won"]


def fitness_of(res):
    """Weight the weakest road heavily, so a pilot can't score well on one road alone.
    Practice flights count a little: enough to keep the ending skill, not enough to dominate."""
    roads = [flight_score(r) for k, r in res.items() if k >= 0 and not r["skipped"]]
    practice = [flight_score(r) for k, r in res.items() if k < 0 and not r["skipped"]]
    score = 0.6 * float(np.mean(roads)) + 0.4 * min(roads) if roads else 0.0
    return score + (0.25 * float(np.mean(practice)) if practice else 0.0)


def rank_fitness(res):
    """Fitness from each pilot's standing on each road, so every road counts the same.

    Summed points let one skill dominate: finishing Road 1 is worth 2500, reaching 75% of Road 2 only
    750, so pilots that had learned Road 2 lost to Road 1 finishers and died out. Here each road (and
    each practice road) ranks the pilots that flew it from 0 (worst) to 1000 (best), ties sharing
    their average rank. A pilot's fitness is 0.6 x its mean standing + 0.4 x its worst standing, plus
    a quarter of its mean practice standing."""
    standing = {gid: {} for gid in res}
    keys = {k for d in res.values() for k, r in d.items() if not r["skipped"]}
    for k in keys:
        flown = {gid: flight_score(d[k]) for gid, d in res.items() if k in d and not d[k]["skipped"]}
        if not flown:
            continue
        ids = list(flown)
        scores = np.array([flown[g] for g in ids])
        order = scores.argsort(kind="stable")
        ranks = np.empty(len(ids))
        ranks[order] = np.arange(len(ids))
        for v in np.unique(scores):                  # ties share their average rank
            same = scores == v
            ranks[same] = ranks[same].mean()
        for gid, rnk in zip(ids, ranks):
            standing[gid][k] = 1000.0 * rnk / max(1, len(ids) - 1)
    fits = {}
    for gid, st in standing.items():
        roads = [v for k, v in st.items() if k >= 0]
        practice = [v for k, v in st.items() if k < 0]
        score = 0.6 * float(np.mean(roads)) + 0.4 * min(roads) if roads else 0.0
        fits[gid] = score + (0.25 * float(np.mean(practice)) if practice else 0.0)
    return fits


def play(a):
    with open(a.play, "rb") as f:
        blob = pickle.load(f)
    cfg = config_from_text(blob["config_text"])
    net = neat.nn.FeedForwardNetwork.create(blob["genome"], cfg)
    pal = live.Palettes()
    win = live.GameWindow(0, live.RoadVision(pal))
    try:
        for road in a.roads:
            print(f"road {road}:", win.fly(net, road), flush=True)
    finally:
        win.close()
        pal.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--pop", type=int, default=40)
    p.add_argument("--gens", type=int, default=200)
    p.add_argument("--roads", type=int, nargs="+", default=[0, 1, 2], help="road numbers, 0 to 29")
    p.add_argument("--seed-from", default="runs/clear_vision")
    p.add_argument("--target-species", type=int, default=5)
    p.add_argument("--practice-roads", type=int, nargs="*", default=[0, 1, 2],
                   help="roads where candidates also practise from where pilots get stuck")
    p.add_argument("--out", default="runs/real")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--play", help="fly a saved pilot on --roads instead of training")
    p.add_argument("--fast", action="store_true",
                   help="run the games fast-forwarded (each window then keeps a CPU core busy)")
    p.add_argument("--link-sim", default="runs/realsim",
                   help="sim run to swap pilots with ('' to switch off)")
    p.add_argument("--nursery-lines", type=int, default=3, help="sim pilots adapting in reserved places at once")
    p.add_argument("--nursery-slots", type=int, default=3, help="places each sim pilot's line keeps")
    p.add_argument("--nursery-gens", type=int, default=8, help="generations a line keeps its places")
    a = p.parse_args()
    if a.play:
        return play(a)

    os.makedirs(a.out, exist_ok=True)
    state = {"gen": 0, "best": -1.0, "roads": list(a.roads)}
    ck = latest_checkpoint(a.out) if a.resume else None
    if ck:
        pop = neat.Checkpointer.restore_checkpoint(ck)
        with open(os.path.join(a.out, "neat.cfg")) as f:
            cfg_text = f.read()
        with open(os.path.join(a.out, "state.pkl"), "rb") as f:
            state = pickle.load(f)
        print(f"resumed from {ck}: generation {state['gen']}, roads {state['roads']}")
        cfg_text = add_gravity_input(pop, cfg_text, os.path.join(a.out, "neat.cfg"))
    else:
        pop, cfg_text = seed_population(a)
        with open(os.path.join(a.out, "neat.cfg"), "w") as f:
            f.write(cfg_text)
        cfg_text = add_gravity_input(pop, cfg_text, os.path.join(a.out, "neat.cfg"))
    pop.config.reproduction_config.min_species_size = 1     # many small species must not stop the run
    if state.get("fitness_mode") != "rank":
        # fitness is now a standing per road, on a different scale from the summed points before:
        # forget each species' old scores, or every species would look stalled and be removed
        for sp in pop.species.species.values():
            sp.fitness_history, sp.fitness, sp.last_improved = [], None, pop.generation
        state["fitness_mode"] = "rank"
        print("fitness: standing on each road (species histories reset)", flush=True)
    pop.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(a.out, "neat-checkpoint-")))

    log_path = os.path.join(a.out, "log.csv")
    new_log = not (ck and os.path.exists(log_path))
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["gen", "roads", "best", "mean", "best_progress", "wins", "species", "secs"])
    palettes = live.Palettes()
    fleet = Fleet(a.windows, palettes, cfg_text, a.fast)
    latest = os.path.join(a.out, "best.pkl")
    if not os.path.exists(latest):
        latest = os.path.join(a.out, "latest_best.pkl")
    if os.path.exists(latest):
        with open(latest, "rb") as f:
            start_carrier = pickle.load(f)["genome"]
        fleet.carriers = {r: start_carrier for r in a.practice_roads}
        fleet.handover = {0: None} if 0 in a.practice_roads else {}   # others are set after a generation
    print(f"{a.windows} windows, population {len(pop.population)}, roads {state['roads']}", flush=True)

    link = {"prev_keys": None}
    nursery = state.get("nursery") or migrate.Nursery(a.nursery_lines, a.nursery_slots, a.nursery_gens)
    nursery.lines, nursery.slots, nursery.gens = a.nursery_lines, a.nursery_slots, a.nursery_gens

    def eval_genomes(genomes, cfg):
        t0 = time.time()
        new_lines = 0
        if a.link_sim and link["prev_keys"] is not None:
            # sim pilots take reserved places among this generation's new offspring (never last
            # generation's survivors) and keep them for a few generations while they adapt
            incoming = migrate.load(os.path.join(a.link_sim, "migrants.pkl"))
            new_lines = nursery.place(genomes, link["prev_keys"], incoming, cfg.genome_config,
                                      pop.reproduction.innovation_tracker)
        res = fleet.evaluate(genomes, state["roads"], a.practice_roads)
        fits = rank_fitness(res)                    # selection: standing on each road
        raw = {gid: fitness_of(res[gid]) for gid, _ in genomes}    # summed points, for the log and best.pkl
        for gid, g in genomes:
            g.fitness = fits[gid]
        best_id = max(fits, key=fits.get)
        link["prev_keys"] = {gid for gid, _ in genomes}
        ranked = sorted(fits, key=fits.get, reverse=True)
        nursery.record(genomes, fits)
        state["nursery"] = nursery
        migrate.export(os.path.join(a.out, "migrants.pkl"), [dict(genomes)[g] for g in ranked[:5]],
                       state["gen"], "real")
        lines = [g for g in nursery.groups if any(k in fits for k in g["keys"])]   # not stale ones after a resume
        import_text = (f"sim lines {len(lines)} (+{new_lines} new; ages "
                       + ",".join(str(g["age"]) for g in lines) + "; best ranks "
                       + ",".join(str(min(ranked.index(k) for k in g["keys"] if k in fits) + 1) for g in lines) + ")"
                       if lines else "sim lines 0")
        champ = dict(genomes)[best_id]
        br = res[best_id]
        wins = sum(r["won"] for d in res.values() for r in d.values())
        n_species = len(pop.species.species) if pop.species.species else 0
        secs = time.time() - t0
        flights = [r for d in res.values() for k, r in d.items() if k >= 0]
        tries = [r for d in res.values() for k, r in d.items() if k < 0]
        ran = [r for r in tries if not r["skipped"]]
        practice_text = (f"practice {len(ran)}/{len(tries)}, driven {sum(r['end_score'] >= 0.8 for r in ran)}, "
                         f"won {sum(r['won'] for r in ran)}, takeover rows "
                         + " ".join(f"r{k}:{v if v is not None else 'end-12'}" for k, v in sorted(fleet.handover.items())))
        hz = np.mean([r["hz"] for r in flights])
        tunnels = sum(r["tunnels"] for r in flights)
        why = Counter(r["why"] for r in flights)
        why_text = ", ".join(f"{k} {v}" for k, v in why.most_common())
        ends = sum(r["reached_end"] for r in flights)
        aimed = sum(r["end_score"] >= 0.8 for r in flights)
        jumping = np.mean([r["jump_frac"] for r in flights])
        detail = " ".join(f"{'r' if k >= 0 else 'p'}{k if k >= 0 else -k - 1}:{v['progress']:.0%}"
                          for k, v in sorted(br.items()))
        furthest = " ".join(f"r{k}:{max(d[k]['progress'] for d in res.values() if k in d):.0%}"
                            for k in sorted(state["roads"]))
        print(f"gen {state['gen']:4d}  best {fits[best_id]:7.1f} ({raw[best_id]:.0f} pts)  mean {np.mean(list(fits.values())):7.1f}  "
              f"[{detail}]  furthest [{furthest}]  wins {wins}  species {n_species}  "
              f"{len(champ.nodes)}n/{sum(c.enabled for c in champ.connections.values())}c  {hz:.0f}Hz  tunnels {tunnels}  [{why_text}]  reached end {ends}  drove in-lane {aimed}  jump held {jumping:.0%}  {practice_text}  {import_text}  {secs:.0f}s",
              flush=True)
        w.writerow([state["gen"], " ".join(map(str, state["roads"])), f"{raw[best_id]:.1f}",
                    f"{np.mean(list(fits.values())):.1f}", detail, wins, n_species, f"{secs:.0f}"])
        log.flush()
        palettes.save()

        blob = {"kind": "neat", "genome": champ, "config_text": cfg_text, "obs_mode": "vision",
                "meta": {"gen": state["gen"], "roads": list(state["roads"]), "fitness": fits[best_id]}}
        with open(os.path.join(a.out, "latest_best.pkl"), "wb") as f:
            pickle.dump(blob, f)
        top_raw = max(raw, key=raw.get)                 # best.pkl: the most points ever, across runs
        if raw[top_raw] > state["best"]:
            state["best"] = raw[top_raw]
            keep = dict(blob, genome=dict(genomes)[top_raw])
            keep["meta"] = dict(blob["meta"], fitness=raw[top_raw])
            with open(os.path.join(a.out, "best.pkl"), "wb") as f:
                pickle.dump(keep, f)

        # a pilot that finishes every current road unlocks the next one
        for road in a.practice_roads:
            # the pilot that got furthest on a road carries next generation's practice flights there,
            # handing over a few rows before where it stopped: that's where the population is stuck
            flown = [gid for gid in fits if road in res[gid]]
            if flown:
                best = max(flown, key=lambda g: res[g][road]["progress"])
                fleet.carriers[road] = dict(genomes)[best]
                r = res[best][road]
                reach = 3 + r["progress"] * (r["length"] - 3)
                fleet.handover[road] = max(3, min(r["length"] - live.HANDOVER_ROWS, int(reach) - 6))
        if all(r["won"] for k, r in br.items() if k >= 0) and max(state["roads"]) < live.ROADS - 1:
            state["roads"].append(max(state["roads"]) + 1)
            state["best"] = -1.0
            print(f"       -> road {state['roads'][-1]} added", flush=True)
        sc = cfg.species_set_config
        # imported pilots can split off many small species; push back harder when far over target
        if n_species > a.target_species:
            step = 0.5 if n_species > 1.5 * a.target_species else 0.1
            sc.compatibility_threshold = min(10.0, sc.compatibility_threshold + step)
        elif n_species and n_species < a.target_species:
            # far under target the population has converged: split it up quickly
            step = 0.3 if n_species <= a.target_species / 2 else 0.1
            sc.compatibility_threshold = max(0.5, sc.compatibility_threshold - step)
        state["gen"] += 1
        with open(os.path.join(a.out, "state.pkl"), "wb") as f:
            pickle.dump(state, f)

    try:
        pop.run(eval_genomes, a.gens)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        fleet.close()
        log.close()
        palettes.save()


if __name__ == "__main__":
    main()
