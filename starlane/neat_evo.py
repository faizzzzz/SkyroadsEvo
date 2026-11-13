"""NEAT (NeuroEvolution of Augmenting Topologies) support, via neat-python.

Like MarIO, every network starts tiny (a few random input->output links, no
hidden neurons). Evolution adds and removes links and neurons, and similar
networks are grouped into species, so new structure gets time to prove itself.
"""
import os
import tempfile

import neat
import numpy as np

from . import config as C
from .level import generate
from .policy import to_actions
from .sim import BatchSim, obs_size

CONFIG_TEMPLATE = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
no_fitness_termination = True
pop_size              = {pop}
reset_on_extinction   = True

[DefaultGenome]
num_inputs            = {n_in}
num_outputs           = 3
num_hidden            = 0
feed_forward          = True
initial_connection    = partial_direct {conn_frac}

activation_default    = tanh
activation_mutate_rate = 0.0
activation_options    = tanh
aggregation_default   = sum
aggregation_mutate_rate = 0.0
aggregation_options   = sum

bias_init_mean        = 0.0
bias_init_stdev       = 0.5
bias_init_type        = gaussian
bias_max_value        = 5.0
bias_min_value        = -5.0
bias_mutate_power     = 0.3
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.05

response_init_mean    = 1.0
response_init_stdev   = 0.0
response_init_type    = gaussian
response_max_value    = 1.0
response_min_value    = 1.0
response_mutate_power = 0.0
response_mutate_rate  = 0.0
response_replace_rate = 0.0

weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_init_type      = gaussian
weight_max_value      = 8.0
weight_min_value      = -8.0
weight_mutate_power   = 0.4
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.05

enabled_default       = True
enabled_mutate_rate   = 0.02
enabled_rate_to_true_add = 0.0
enabled_rate_to_false_add = 0.0

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob         = 0.4
conn_delete_prob      = 0.2
node_add_prob         = 0.15
node_delete_prob      = 0.05
single_structural_mutation = False
structural_mutation_surer  = default

[DefaultSpeciesSet]
compatibility_threshold = {compat}

[DefaultStagnation]
species_fitness_func  = max
max_stagnation        = 25
species_elitism       = 2

[DefaultReproduction]
elitism               = 2
survival_threshold    = 0.2
min_species_size      = 1
"""


def make_config_text(obs_mode: str, pop: int, conn_frac: float = 0.03, compat: float = 3.0) -> str:
    return CONFIG_TEMPLATE.format(pop=pop, n_in=obs_size(obs_mode), conn_frac=conn_frac,
                                  compat=compat)


_CFG_CACHE: dict[str, neat.Config] = {}


def config_from_text(text: str) -> neat.Config:
    if text not in _CFG_CACHE:
        fd, path = tempfile.mkstemp(suffix=".cfg")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        try:
            _CFG_CACHE[text] = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                           neat.DefaultSpeciesSet, neat.DefaultStagnation, path)
        finally:
            os.remove(path)
    return _CFG_CACHE[text]


class NeatPolicy:
    """A batch of NEAT networks with the same act() interface as the MLP population."""

    def __init__(self, genomes, config):
        self.nets = [neat.nn.FeedForwardNetwork.create(g, config) for g in genomes]

    def act(self, obs: np.ndarray, alive: np.ndarray | None = None) -> np.ndarray:
        out = np.zeros((len(self.nets), 3), np.float32)
        idx = range(len(self.nets)) if alive is None else np.nonzero(alive)[0]
        for i in idx:
            out[i] = self.nets[i].activate(obs[i].tolist())
        return out


def run_level_neat(genomes, cfg_text, obs_mode, seed, difficulty, obs_noise=0.0):
    """Fly every genome on one level. Returns (fitness, reasons)."""
    policy = NeatPolicy(genomes, config_from_text(cfg_text))
    level = generate(seed, difficulty)
    sim = BatchSim(level, len(genomes), obs_mode, obs_noise, np.random.default_rng(seed))
    while sim.alive.any():
        alive = sim.alive
        out = policy.act(sim.observe(), alive)
        s, t, j = to_actions(out)
        sim.step(np.where(alive, s, 0), np.where(alive, t, 0), alive & j)
    return sim.fitness(), sim.reason.copy()


def _job(args):
    return run_level_neat(*args)


def evaluate_neat(pool, genomes, cfg_text, obs_mode, seeds, difficulty, noise):
    jobs = [(genomes, cfg_text, obs_mode, s, difficulty, noise) for s in seeds]
    results = list(pool.map(_job, jobs)) if pool else [run_level_neat(*j) for j in jobs]
    fit = np.mean([r[0] for r in results], 0)
    wins = np.mean([r[1] == C.WIN for r in results], 0)
    reasons = np.stack([r[1] for r in results], 1)
    return fit, wins, reasons
