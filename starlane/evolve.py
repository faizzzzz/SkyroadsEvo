"""Genetic algorithm with self-adaptive mutation for neural network policies."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config as C
from .level import generate
from .policy import MLPSpec, to_actions
from .sim import BatchSim


def run_level(pop, sizes, obs_mode, seed, difficulty, obs_noise=0.0):
    """Evaluate every genome in pop on one level. Returns (fitness, reasons)."""
    spec = MLPSpec.from_sizes(sizes)
    level = generate(seed, difficulty)
    sim = BatchSim(level, pop.shape[0], obs_mode, obs_noise, np.random.default_rng(seed))
    while sim.alive.any():
        alive = sim.alive
        obs = sim.observe()
        steer = np.zeros(sim.n, np.int8)
        thr = np.zeros(sim.n, np.int8)
        jmp = np.zeros(sim.n, bool)
        out = spec.forward(pop[alive], obs[alive])
        s, t, j = to_actions(out)
        steer[alive], thr[alive], jmp[alive] = s, t, j
        sim.step(steer, thr, jmp)
    return sim.fitness(), sim.reason.copy()


def _run_level_args(args):
    return run_level(*args)


class Evaluator:
    def __init__(self, workers: int = 1):
        self.pool = ProcessPoolExecutor(workers) if workers > 1 else None

    def evaluate(self, pop, spec, obs_mode, seeds, difficulty, obs_noise=0.0):
        """Mean fitness and win rate over several levels."""
        jobs = [(pop, spec.sizes, obs_mode, s, difficulty, obs_noise) for s in seeds]
        if self.pool:
            results = list(self.pool.map(_run_level_args, jobs))
        else:
            results = [run_level(*j) for j in jobs]
        fit = np.mean([r[0] for r in results], 0)
        wins = np.mean([r[1] == C.WIN for r in results], 0)
        reasons = np.stack([r[1] for r in results], 1)
        return fit, wins, reasons

    def close(self):
        if self.pool:
            self.pool.shutdown()


class GA:
    def __init__(self, spec: MLPSpec, pop_size=200, elite=8, tournament=5,
                 crossover=0.3, sigma0=0.08, sigma_min=0.02, seed=0):
        self.spec = spec
        self.sigma_min = sigma_min
        self.rng = np.random.default_rng(seed)
        self.n = pop_size
        self.elite = elite
        self.k = tournament
        self.p_cross = crossover
        self.pop = spec.init(pop_size, self.rng)
        self.sigma = np.full(pop_size, sigma0, np.float32)
        self.groups = spec.neuron_groups()
        self.n_groups = self.groups.max() + 1
        self.tau = 1.0 / np.sqrt(2 * np.sqrt(spec.dim))

    def _select(self, fit):
        idx = self.rng.integers(0, self.n, (self.k,))
        return idx[np.argmax(fit[idx])]

    def ask(self) -> np.ndarray:
        return self.pop

    def champion(self, fit, wins):
        """(genome to save, its index, win-rate statistic used for the curriculum)."""
        order = np.argsort(-fit)
        return self.pop[order[0]], order[0], float(wins[order[:10]].mean())

    @property
    def step_size(self):
        return float(self.sigma.mean())

    def state(self):
        return {"pop": self.pop, "sigma": self.sigma}

    def load(self, f):
        self.pop, self.sigma = f["pop"], f["sigma"]

    def tell(self, fit: np.ndarray):
        """Produce the next generation from fitness of the current one."""
        order = np.argsort(-fit)
        new_pop = np.empty_like(self.pop)
        new_sig = np.empty_like(self.sigma)
        new_pop[:self.elite] = self.pop[order[:self.elite]]
        new_sig[:self.elite] = self.sigma[order[:self.elite]]
        for i in range(self.elite, self.n):
            a = self._select(fit)
            child = self.pop[a].copy()
            sig = self.sigma[a]
            if self.rng.random() < self.p_cross:
                b = self._select(fit)
                take = self.rng.random(self.n_groups) < 0.5
                mask = take[self.groups]
                child[mask] = self.pop[b][mask]
                sig = np.sqrt(sig * self.sigma[b])
            sig = float(np.clip(sig * np.exp(self.tau * self.rng.normal()), self.sigma_min, 0.5))
            child += sig * self.rng.normal(0, 1, child.shape).astype(np.float32)
            new_pop[i] = child
            new_sig[i] = sig
        self.pop, self.sigma = new_pop, new_sig


class ES:
    """OpenAI-style evolution strategy with antithetic sampling, centred-rank
    fitness shaping and Adam. The population is the current mean genome plus
    pairs of mirrored Gaussian perturbations around it; index 0 is the mean
    itself, so its fitness is measured on the same levels every generation."""

    def __init__(self, spec: MLPSpec, pop_size=200, sigma=0.05, lr=0.03,
                 weight_decay=0.005, seed=0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.half = max(1, (pop_size - 1) // 2)
        self.sigma = sigma
        self.lr = lr
        self.wd = weight_decay
        self.theta = spec.init(1, self.rng)[0]
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        self.eps = None

    def ask(self) -> np.ndarray:
        self.eps = self.rng.normal(0, 1, (self.half, self.theta.size)).astype(np.float32)
        return np.concatenate([self.theta[None],
                               self.theta + self.sigma * self.eps,
                               self.theta - self.sigma * self.eps]).astype(np.float32)

    def champion(self, fit, wins):
        return self.theta.copy(), 0, float(wins[0])

    @property
    def step_size(self):
        return self.sigma

    def state(self):
        return {"theta": self.theta, "m": self.m, "v": self.v, "t": self.t}

    def load(self, f):
        self.theta, self.m, self.v, self.t = f["theta"], f["m"], f["v"], int(f["t"])

    def tell(self, fit: np.ndarray):
        f = fit[1:]
        ranks = np.empty(f.size)
        ranks[np.argsort(f)] = np.arange(f.size)
        shaped = ranks / (f.size - 1) - 0.5
        pos, neg = shaped[:self.half], shaped[self.half:]
        grad = ((pos - neg) @ self.eps) / (2 * self.half * self.sigma)
        grad = grad - self.wd * self.theta           # ascent direction
        self.t += 1
        b1, b2 = 0.9, 0.999
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad ** 2
        mh = self.m / (1 - b1 ** self.t)
        vh = self.v / (1 - b2 ** self.t)
        self.theta = (self.theta + self.lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
