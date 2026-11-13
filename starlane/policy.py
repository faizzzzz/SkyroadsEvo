"""Fixed-topology neural network policies stored as flat genome vectors.

A population is a (N, D) matrix; forward() runs all N networks in one batch.
"""
import numpy as np

N_OUT = 3  # steer, throttle, jump


class MLPSpec:
    def __init__(self, n_in: int, hidden=(32, 16)):
        self.sizes = [n_in, *hidden, N_OUT]
        self.shapes = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.shapes.append((a, b))   # weight
            self.shapes.append((b,))     # bias
        self.dim = sum(int(np.prod(s)) for s in self.shapes)

    def to_dict(self):
        return {"sizes": self.sizes}

    @classmethod
    def from_sizes(cls, sizes):
        return cls(sizes[0], tuple(sizes[1:-1]))

    def init(self, n: int, rng: np.random.Generator) -> np.ndarray:
        parts = []
        for s in self.shapes:
            if len(s) == 2:
                parts.append(rng.normal(0, 1.0 / np.sqrt(s[0]), (n, s[0] * s[1])))
            else:
                parts.append(np.zeros((n, s[0])))
        return np.concatenate(parts, 1).astype(np.float32)

    def unpack(self, pop: np.ndarray):
        n = pop.shape[0]
        out, i = [], 0
        for s in self.shapes:
            k = int(np.prod(s))
            out.append(pop[:, i:i + k].reshape(n, *s))
            i += k
        return out

    def neuron_groups(self) -> np.ndarray:
        """Integer label per gene: genes feeding the same neuron share a label.
        Used for crossover that swaps whole neurons rather than single weights."""
        labels, i, base = np.zeros(self.dim, np.int64), 0, 0
        for s in self.shapes:
            k = int(np.prod(s))
            if len(s) == 2:
                labels[i:i + k] = base + np.tile(np.arange(s[1]), s[0])
            else:
                labels[i:i + k] = base + np.arange(s[0])
                base += s[0]
            i += k
        return labels

    def forward(self, pop: np.ndarray, obs: np.ndarray) -> np.ndarray:
        params = self.unpack(pop)
        h = obs
        for li in range(0, len(params), 2):
            W, b = params[li], params[li + 1]
            h = np.einsum("ni,nij->nj", h, W, optimize=True) + b
            h = np.tanh(h)
        return h  # (N, 3) in [-1, 1]


def to_actions(out: np.ndarray):
    steer = np.where(out[:, 0] > 0.33, 1, np.where(out[:, 0] < -0.33, -1, 0))
    throttle = np.where(out[:, 1] > 0.33, 1, np.where(out[:, 1] < -0.33, -1, 0))
    jump = out[:, 2] > 0.5
    return steer, throttle, jump


def save_genome(path, genome: np.ndarray, spec: MLPSpec, obs_mode: str, meta: dict | None = None):
    np.savez(path, genome=genome, sizes=np.array(spec.sizes), obs_mode=obs_mode,
             meta=np.array(repr(meta or {})))


def load_genome(path):
    f = np.load(path, allow_pickle=False)
    spec = MLPSpec.from_sizes([int(s) for s in f["sizes"]])
    mode = str(f["obs_mode"])
    from .sim import obs_size
    if spec.sizes[0] != obs_size(mode):
        raise ValueError(f"{path} expects {spec.sizes[0]} inputs but the '{mode}' observation now "
                         f"has {obs_size(mode)}; it was trained with an older observation layout")
    return f["genome"].astype(np.float32), spec, mode
