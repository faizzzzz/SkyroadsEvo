"""Procedurally generated roads in the game's cell encoding, for the matched simulator.

These are original layouts built from the same kinds of pieces the real game uses (holes,
narrow strips, tunnels, low blocks, tall walls, deadly tiles, and a final tunnel that must be
entered at road level). No real road is copied. Only overall statistics measured across the real
roads guide the mix: lengths of about 50-125 rows, about half of all cells empty, very frequent
1-3 lane strips, gaps almost always one row long, many low blocks, deadly tiles on some roads, and
lower gravity on about one road in ten.
"""
import numpy as np

FLOOR, TUNNEL, LOW, TALL, DEADLY, HOLE = 0x0003, 0x0103, 0x0203, 0x0403, 0x000C, 0
LANES = 7
START_ROW = 3
PAD_ROWS = 40


class Road:
    def __init__(self, cells, length, gravity, tunnel_lane, seed, difficulty):
        self.cells = cells                  # (length + PAD_ROWS, 7) uint16
        self.length = length                # rows; the final tunnel sits on row length - 1
        self.gravity = gravity              # units / s^2
        self.tunnel_lanes = {tunnel_lane}
        self.seed = seed
        self.difficulty = difficulty


class _Builder:
    def __init__(self, rng, d):
        self.rng, self.d = rng, d
        self.rows = []
        self.safe = 3

    def row(self, cells):
        self.rows.append(np.array(cells, np.uint16))

    def drift(self, p=0.4):
        if self.rng.random() < p:
            self.safe = int(np.clip(self.safe + self.rng.choice([-1, 1]), 0, LANES - 1))

    def strip(self, lo, hi, cell=FLOOR):
        r = [HOLE] * LANES
        for j in range(lo, hi + 1):
            r[j] = cell
        return r

    def wide(self, n):
        width = int(self.rng.integers(2, 8))
        lo = int(np.clip(self.safe - self.rng.integers(0, width), 0, LANES - width))
        for _ in range(n):
            self.row(self.strip(lo, lo + width - 1))

    def narrow(self, n):
        for i in range(n):
            if i % 3 == 0:
                prev = self.safe
                self.drift(0.6)
                if self.safe != prev:               # overlap row so the lane change is drivable
                    self.row(self.strip(min(prev, self.safe), max(prev, self.safe)))
            self.row(self.strip(self.safe, self.safe))

    def gap(self):
        self.wide(int(self.rng.integers(4, 7)))
        g = 1 + int(self.rng.random() < 0.08 + 0.15 * self.d)         # real gaps: almost always one row
        for _ in range(g):
            self.row([HOLE] * LANES)
        self.wide(3)

    def gap_shift(self):
        """A gap reached along a narrow strip, landing on road set off to one side, often with blocks
        soon after: jump, steer in the air, then deal with what's next. A combination real pilots
        die on; the layout itself is random."""
        rng = self.rng
        width = int(rng.integers(1, 3))
        lo = int(np.clip(self.safe, 0, LANES - width))
        for _ in range(int(rng.integers(3, 7))):
            self.row(self.strip(lo, lo + width - 1))
        for _ in range(1 + int(rng.random() < 0.1 + 0.15 * self.d)):
            self.row([HOLE] * LANES)
        reach = 2 if self.d > 0.4 and rng.random() < 0.5 else 1
        shift = int(rng.choice([-reach, reach])) if rng.random() < 0.85 else 0
        land_w = int(rng.integers(1, 4))
        land_lo = int(np.clip(lo + shift if shift > 0 else lo + shift - land_w + 1, 0, LANES - land_w))
        for _ in range(int(rng.integers(2, 5))):
            self.row(self.strip(land_lo, land_lo + land_w - 1))
        self.safe = int(rng.integers(land_lo, land_lo + land_w))
        if rng.random() < 0.5 + 0.3 * self.d:          # blocks right after the landing, one lane open
            wide_lo, wide_hi = max(0, land_lo - 1), min(LANES - 1, land_lo + land_w)
            r = self.strip(wide_lo, wide_hi, LOW)
            self.drift(0.6)
            self.safe = int(np.clip(self.safe, wide_lo, wide_hi))
            r[self.safe] = FLOOR
            self.row(self.strip(wide_lo, wide_hi))
            self.row(r)
            self.row(self.strip(wide_lo, wide_hi))

    def holes(self, n):
        p = 0.3 + 0.4 * self.d
        for i in range(n):
            if i % 3 == 0:
                self.drift(0.5)
            r = [HOLE if self.rng.random() < p else FLOOR for _ in range(LANES)]
            r[self.safe] = FLOOR
            self.row(r)

    def tunnels(self):
        self.wide(3)
        n = int(self.rng.integers(2, 5))
        lanes = {self.safe} | {j for j in range(LANES) if self.rng.random() < 0.3}
        for _ in range(n):
            r = [HOLE] * LANES
            for j in lanes:
                r[j] = TUNNEL
            self.row(r)
        self.wide(2)

    def low_blocks(self):
        self.wide(4)
        r = self.strip(0, LANES - 1, LOW)
        for _ in range(int(self.rng.integers(1, 3))):
            self.row(r)
        self.wide(3)

    def walls(self):
        self.wide(4)
        width = 1 if self.d > 0.6 else 2
        lo = int(np.clip(self.safe - (width - 1) * int(self.rng.random() < 0.5), 0, LANES - width))
        for _ in range(int(self.rng.integers(1, 3))):
            r = [TALL] * LANES
            for j in range(lo, lo + width):
                r[j] = FLOOR
            self.row(r)
        self.safe = lo
        self.wide(3)

    def scattered_blocks(self, n):
        """Low blocks dotted across a strip; sometimes one sits in the open lane and must be jumped."""
        width = int(self.rng.integers(2, 6))
        lo = int(np.clip(self.safe - self.rng.integers(0, width), 0, LANES - width))
        for i in range(n):
            r = self.strip(lo, lo + width - 1)
            for j in range(lo, lo + width):
                if j != self.safe and self.rng.random() < 0.35 + 0.3 * self.d:
                    r[j] = LOW
            if self.rng.random() < 0.12 + 0.2 * self.d:
                r[self.safe] = LOW
            self.row(r)

    def deadly(self, n):
        p = 0.3 + 0.4 * self.d
        for i in range(n):
            if i % 3 == 0:
                self.drift(0.5)
            r = [DEADLY if self.rng.random() < p else FLOOR for _ in range(LANES)]
            r[self.safe] = FLOOR
            self.row(r)

    def finish(self):
        """A run-in that narrows to one lane, then the final tunnel on the last row."""
        self.wide(3)
        n = int(self.rng.integers(5, 10))
        for _ in range(n):
            self.row(self.strip(self.safe, self.safe))
        r = [HOLE] * LANES
        r[self.safe] = TUNNEL
        self.row(r)


def gauntlet(seed: int) -> Road:
    """A long, dense course at full difficulty with every kind of challenge, twice over, in random
    order and with only a row of breathing space between them: single-lane strips, gaps (including
    ones with a sideways landing), holes, tunnels, tall walls, low blocks, scattered blocks and
    deadly tiles. Original random layouts, like every road here."""
    rng = np.random.default_rng(seed)
    b = _Builder(rng, 1.0)
    for _ in range(START_ROW + 1):
        b.row(b.strip(1, 5))
    b.safe = int(rng.integers(2, 5))
    n = lambda: int(rng.integers(4, 9))
    pieces = [lambda: b.narrow(n()), b.gap_shift, b.gap, lambda: b.holes(n()), b.tunnels, b.walls,
              lambda: b.deadly(n()), lambda: b.scattered_blocks(n()), b.low_blocks]
    order = [p for p in pieces for _ in range(2)]
    rng.shuffle(order)
    for piece in order:
        piece()
        b.wide(1)
    b.finish()
    length = len(b.rows)
    cells = np.zeros((length + PAD_ROWS, LANES), np.uint16)
    cells[:length] = np.stack(b.rows)
    gravity = 148000.0 if rng.random() < 0.9 else 128000.0
    return Road(cells, length, gravity, b.safe, seed, 1.0)


def generate(seed: int, difficulty: float = 0.5) -> Road:
    d = float(np.clip(difficulty, 0, 1))
    rng = np.random.default_rng(seed)
    b = _Builder(rng, d)
    # the ship starts on row 3 in lane 3; hazards can begin right after it, and the open lane
    # isn't always the one in front of the ship, so a pilot must react from a standstill
    for _ in range(START_ROW + 1):
        b.row(b.strip(1, 5))
    b.safe = int(rng.integers(2, 5)) if rng.random() < 0.6 else 3
    if rng.random() < 0.5:
        b.wide(int(rng.integers(1, 4)))
    if rng.random() < 0.3 * d:                    # sometimes the hard part comes straight away
        b.gap_shift()
    target = int(45 + 60 * d + rng.integers(0, 25))
    kinds = ["wide", "narrow", "gap", "holes", "tunnels", "low", "walls", "deadly", "blocks", "gapshift"]
    w = np.array([1.6 - 0.8 * d, 1.0 + 1.0 * d, 1.0 + 0.6 * d, 0.6 + 0.8 * d,
                  0.6, 0.4 + 0.4 * d, 0.3 + 0.6 * d, 0.2 + 0.8 * d, 0.8 + 0.8 * d, 0.4 + 1.0 * d])
    w /= w.sum()
    while len(b.rows) < target:
        k = rng.choice(kinds, p=w)
        n = int(rng.integers(4, 12))
        {"wide": lambda: b.wide(n), "narrow": lambda: b.narrow(n), "gap": b.gap, "holes": lambda: b.holes(n),
         "tunnels": b.tunnels, "low": b.low_blocks, "walls": b.walls, "deadly": lambda: b.deadly(n),
         "blocks": lambda: b.scattered_blocks(n), "gapshift": b.gap_shift}[k]()
        b.wide(int(rng.integers(1, 3 + int(4 * (1 - d)))))
    b.finish()
    length = len(b.rows)
    cells = np.zeros((length + PAD_ROWS, LANES), np.uint16)
    cells[:length] = np.stack(b.rows)
    gravity = 148000.0 if rng.random() < 0.9 else 128000.0   # measured: about one road in ten is lighter
    return Road(cells, length, gravity, b.safe, seed, d)
