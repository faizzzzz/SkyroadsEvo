"""Move pilots between two NEAT runs (the matched simulator and the real game).

Each run numbers its hidden neurons and gene innovations independently, so an incoming
pilot is renumbered with the receiving run's counters before it is written into one of
that run's genomes. Files are written atomically, so a reader never sees half a file.
"""
import copy
import os
import pickle
import random


def export(path, genomes, gen, source):
    """genomes: best first. Saves deep copies."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"gen": gen, "source": source, "genomes": [copy.deepcopy(g) for g in genomes]}, f)
    os.replace(tmp, path)


def load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError):
        return None


def implant(target, source, genome_config, tracker):
    """Make `target` (a genome of the receiving run) a copy of `source` (from the other run)."""
    outputs = set(genome_config.output_keys)
    remap = {}
    new_nodes = {}
    for key, node in source.nodes.items():
        if key in outputs:
            new_key = key
        else:                                        # hidden neuron: give it a fresh number here
            new_key = genome_config.get_new_node_key(new_nodes)
        remap[key] = new_key
        n = copy.deepcopy(node)
        n.key = new_key
        new_nodes[new_key] = n
    new_conns = {}
    for (i, o), cg in source.connections.items():
        ni, no = remap.get(i, i), remap.get(o, o)    # inputs (negative keys) are shared by both runs
        c = copy.deepcopy(cg)
        c.key = (ni, no)
        c.innovation = tracker.get_innovation_number(ni, no, "add_connection")
        new_conns[(ni, no)] = c
    target.nodes = new_nodes
    target.connections = new_conns
    target.fitness = None


def _copy_into(target, source):
    target.nodes = copy.deepcopy(source.nodes)
    target.connections = copy.deepcopy(source.connections)
    target.fitness = None


class Nursery:
    """Reserved places for pilots brought in from the other run.

    A newcomer starts behind a population tuned to the real roads and used to be dropped after one
    flight. Here each newcomer starts a line that keeps `slots` places for `gens` generations: its
    best member from the last generation flies again, the other places get mutated copies of it.
    Afterwards the line's members stay as ordinary genomes and compete normally.
    """

    def __init__(self, lines=3, slots=3, gens=8):
        self.lines, self.slots, self.gens = lines, slots, gens
        self.groups = []             # dicts: source, age, best (genome copy), keys (this generation)
        self.seen = set()            # (source run, source genome key) already taken in

    def _ensure_indexer(self, population_items, genome_config):
        if genome_config.node_indexer is None:       # after a restart: continue above every neuron in use
            from itertools import count
            top = max((k for _, g in population_items for k in g.nodes), default=len(genome_config.output_keys))
            genome_config.node_indexer = count(top + 1)

    def place(self, population_items, protected_keys, incoming, genome_config, tracker, rng=random):
        """Write the nursery into this generation's new offspring. Returns the number of new lines."""
        self._ensure_indexer(population_items, genome_config)
        genome_config.innovation_tracker = tracker
        self.groups = [g for g in self.groups if g["age"] < self.gens]
        spots = [gid for gid, _ in population_items if gid not in protected_keys]
        rng.shuffle(spots)
        by_key = dict(population_items)
        new = 0
        if incoming:
            for src in incoming["genomes"]:
                if len(self.groups) >= self.lines:
                    break
                tag = (incoming.get("source"), src.key)
                if tag in self.seen:
                    continue
                self.seen.add(tag)
                template = copy.deepcopy(by_key[spots[0]] if spots else next(iter(by_key.values())))
                implant(template, src, genome_config, tracker)
                self.groups.append({"source": tag, "age": 0, "best": template, "keys": []})
                new += 1
        for group in self.groups:
            group["keys"] = []
            for i in range(self.slots):
                if not spots:
                    break
                gid = spots.pop()
                _copy_into(by_key[gid], group["best"])
                if i:                                # the first place re-flies the line's best as it is
                    by_key[gid].mutate(genome_config)
                group["keys"].append(gid)
        return new

    def record(self, genomes, fits):
        """After evaluation: each line keeps its best member of this generation."""
        by_key = dict(genomes)
        for group in self.groups:
            flown = [k for k in group["keys"] if k in fits]
            if flown:
                best = max(flown, key=fits.get)
                group["best"] = copy.deepcopy(by_key[best])
                group["best_fit"] = fits[best]
            group["age"] += 1

    def keys(self):
        return {k for g in self.groups for k in g["keys"]}


def bring_in(population_items, protected_keys, incoming, genome_config, tracker, k, rng=random):
    """Overwrite up to k unprotected genomes with the incoming pilots. Returns the overwritten keys."""
    if genome_config.node_indexer is None:           # after a restart: continue above every neuron in use
        from itertools import count
        top = max((k for _, g in population_items for k in g.nodes), default=len(genome_config.output_keys))
        genome_config.node_indexer = count(top + 1)
    spots = [gid for gid, _ in population_items if gid not in protected_keys]
    rng.shuffle(spots)
    by_key = dict(population_items)
    used = []
    for gid, src in zip(spots, incoming[:k]):
        implant(by_key[gid], src, genome_config, tracker)
        used.append(gid)
    return used
