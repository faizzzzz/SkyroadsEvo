"""Load any saved pilot: an MLP genome (.npz) or a NEAT genome (.pkl)."""
import pickle

import numpy as np

from .policy import load_genome


class Pilot:
    """A single policy. act(obs) takes (N, D) observations, returns (N, 3) outputs."""

    def __init__(self, path: str):
        self.path = path
        if path.endswith(".pkl"):
            from .neat_evo import NeatPolicy, config_from_text
            with open(path, "rb") as f:
                blob = pickle.load(f)
            self.kind = "neat"
            self.obs_mode = blob["obs_mode"]
            self.meta = blob.get("meta", {})
            self.genome = blob["genome"]
            from .sim import obs_size
            n_in = config_from_text(blob["config_text"]).genome_config.num_inputs
            if n_in != obs_size(self.obs_mode):
                raise ValueError(f"{path} expects {n_in} inputs but the '{self.obs_mode}' observation "
                                 f"now has {obs_size(self.obs_mode)}; it was trained with an older "
                                 "observation layout")
            self._policy = NeatPolicy([blob["genome"]], config_from_text(blob["config_text"]))
        else:
            genome, spec, obs_mode = load_genome(path)
            self.kind = "mlp"
            self.obs_mode = obs_mode
            self.meta = {}
            self._genome, self._spec = genome[None], spec

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.kind == "neat":
            return np.concatenate([self._policy.act(o[None]) for o in obs])
        return self._spec.forward(np.repeat(self._genome, len(obs), 0), obs)

    def describe(self) -> str:
        if self.kind == "neat":
            n = len(self.genome.nodes)
            c = sum(1 for x in self.genome.connections.values() if x.enabled)
            return f"NEAT, {n} neurons, {c} links, obs={self.obs_mode}"
        return f"MLP {self._spec.sizes}, obs={self.obs_mode}"
