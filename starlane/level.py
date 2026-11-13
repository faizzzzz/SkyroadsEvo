"""Procedurally generated roads for the Starlane simulator.

A level is a grid of rows x lanes. Each cell has a surface height (GAP for a
hole, 0 for floor, BLOCK_H * n for raised blocks) and a tile kind.

The generator keeps a "safe lane" that drifts slowly across the road; every
segment leaves that lane (or a jumpable route over it) passable, so levels are
meant to be beatable with good play.
"""
from dataclasses import dataclass

import numpy as np

from .config import (LANES, BLOCK_H, TALL_H, GAP, K_NORMAL, K_BOOST, K_STICKY,
                     K_BURN, K_FUEL, V_MAX)

PAD_COLS = 4      # gap columns added on each side so off-road lookups are holes
PAD_ROWS = 60     # floor rows added past the finish line


@dataclass
class Level:
    top: np.ndarray        # (rows + PAD_ROWS, LANES + 2*PAD_COLS) float32 surface heights
    kind: np.ndarray       # same shape, int8 tile kinds
    length: int            # finish line (row index)
    fuel_rows: float       # rows of travel a full tank lasts
    oxygen_secs: float     # seconds of oxygen
    seed: int = 0
    difficulty: float = 0.0

    @property
    def max_seconds(self) -> float:
        return self.length / (V_MAX * 0.2) + 10.0


class _Builder:
    def __init__(self, rng: np.random.Generator, d: float):
        self.rng = rng
        self.d = d
        self.rows_h: list[np.ndarray] = []
        self.rows_k: list[np.ndarray] = []
        self.safe = LANES // 2
        self.recent: list[int] = []   # safe lanes of the last few rows

    def keep_clear(self, h, k=None, hold=4):
        """Clear the safe lane and every safe lane of the last `hold` rows, so a
        lane change always has a few rows of overlap to steer through."""
        self.recent = (self.recent + [self.safe])[-hold:]
        for j in set(self.recent):
            h[j] = 0
            if k is not None:
                k[j] = K_NORMAL

    def row(self, h=None, k=None):
        self.rows_h.append(np.zeros(LANES, np.float32) if h is None else h.astype(np.float32))
        self.rows_k.append(np.zeros(LANES, np.int8) if k is None else k.astype(np.int8))

    def drift(self, p=0.25, lo=0, hi=LANES - 1):
        if self.rng.random() < p:
            self.safe = int(np.clip(self.safe + self.rng.choice([-1, 1]), lo, hi))

    # ---- segments -------------------------------------------------------
    def straight(self, n):
        for _ in range(n):
            self.row()

    def narrow(self, n):
        """Road narrows to a 1-3 lane strip that wanders."""
        width = int(self.rng.integers(1, 4)) if self.d > 0.3 else int(self.rng.integers(2, 4))
        for i in range(n):
            if i % 3 == 0:
                self.drift(0.5, 0, LANES - width)
            lo = min(self.safe, LANES - width)
            h = np.full(LANES, GAP, np.float32)
            h[lo:lo + width] = 0
            self.row(h)

    def gap(self):
        g = int(self.rng.integers(1, 2 + int(1 + 5 * self.d)))
        self.straight(int(self.rng.integers(5, 9)))  # run-up
        for _ in range(g):
            self.row(np.full(LANES, GAP))
        self.straight(3)

    def holes(self, n):
        """Random holes everywhere except a wandering safe lane."""
        p = 0.25 + 0.45 * self.d
        for i in range(n):
            if i % 3 == 0:
                self.drift(0.6)
            h = np.where(self.rng.random(LANES) < p, GAP, 0).astype(np.float32)
            self.keep_clear(h)
            self.row(h)

    def low_blocks(self):
        """A full-width raised slab that must be jumped onto."""
        self.straight(int(self.rng.integers(5, 8)))
        n = int(self.rng.integers(3, 10))
        for _ in range(n):
            self.row(np.full(LANES, BLOCK_H))
        if self.rng.random() < 0.5 * self.d:  # slab ends in a gap
            for _ in range(int(self.rng.integers(1, 3))):
                self.row(np.full(LANES, GAP))
        self.straight(3)

    def walls(self, n):
        """Tall walls across part of the road, leaving an opening at the safe lane."""
        self.straight(4)
        width = 1 if self.d > 0.6 else 2
        self.drift(0.8, 0, LANES - width)
        lo = min(self.safe, LANES - width)
        for _ in range(n):
            h = np.full(LANES, TALL_H, np.float32)
            h[lo:lo + width] = 0
            self.row(h)
        self.straight(3)

    def pillars(self, n):
        """Scattered single blocks (low or tall) to steer around or hop over."""
        p = 0.12 + 0.2 * self.d
        for i in range(n):
            if i % 4 == 0:
                self.drift(0.5)
            h = np.zeros(LANES, np.float32)
            r = self.rng.random(LANES)
            h[r < p] = TALL_H
            h[(r >= p) & (r < 1.5 * p)] = BLOCK_H
            self.keep_clear(h)
            self.row(h)

    def burn(self, n):
        """Burning tiles on part of the road."""
        p = 0.3 + 0.4 * self.d
        for i in range(n):
            if i % 3 == 0:
                self.drift(0.5)
            k = np.where(self.rng.random(LANES) < p, K_BURN, K_NORMAL).astype(np.int8)
            self.keep_clear(np.zeros(LANES), k)
            self.row(None, k)

    def special_strip(self, kind, n):
        lanes = self.rng.random(LANES) < 0.5
        lanes[self.safe] = True
        for _ in range(n):
            self.row(None, np.where(lanes, kind, K_NORMAL))


def generate(seed: int, difficulty: float = 0.5, length: int | None = None) -> Level:
    """Build a level. difficulty in [0, 1]."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    if length is None:
        length = int(180 + 320 * d + rng.integers(0, 80))
    b = _Builder(rng, d)
    b.straight(15)

    segs = ["straight", "gap", "holes", "low", "walls", "pillars", "narrow",
            "burn", "boost", "sticky", "fuel"]
    w = np.array([3.0 - 2 * d, 1.0 + 2 * d, 0.5 + 2 * d, 0.6 + 1.2 * d, 0.4 + 1.5 * d,
                  0.4 + 1.6 * d, 0.3 + 1.5 * d, 0.2 + 1.2 * d, 0.6, 0.2 + 0.6 * d, 0.35])
    w /= w.sum()
    while len(b.rows_h) < length:
        s = rng.choice(segs, p=w)
        n = int(rng.integers(6, 18))
        if s == "straight":
            b.straight(n)
        elif s == "gap":
            b.gap()
        elif s == "holes":
            b.holes(n)
        elif s == "low":
            b.low_blocks()
        elif s == "walls":
            b.walls(int(rng.integers(2, 5)))
        elif s == "pillars":
            b.pillars(n)
        elif s == "narrow":
            b.narrow(n)
        elif s == "burn":
            b.burn(n)
        elif s == "boost":
            b.special_strip(K_BOOST, int(rng.integers(1, 3)))
        elif s == "sticky":
            b.special_strip(K_STICKY, int(rng.integers(3, 8)))
        elif s == "fuel":
            b.special_strip(K_FUEL, 2)
        # breathing room between hazards shrinks with difficulty
        b.straight(int(rng.integers(1, 3 + int(6 * (1 - d)))))

    length = len(b.rows_h)
    for _ in range(PAD_ROWS):
        b.row()
    h = np.stack(b.rows_h)
    k = np.stack(b.rows_k)
    top = np.full((h.shape[0], LANES + 2 * PAD_COLS), GAP, np.float32)
    kind = np.zeros_like(top, dtype=np.int8)
    top[:, PAD_COLS:PAD_COLS + LANES] = h
    kind[:, PAD_COLS:PAD_COLS + LANES] = k
    kind[top <= GAP + 1] = K_NORMAL

    fuel_rows = length * (2.2 - 0.8 * d)
    oxygen_secs = length / (V_MAX * (0.30 + 0.25 * d))
    return Level(top, kind, length, fuel_rows, oxygen_secs, seed, d)
