"""Vectorised Starlane physics: N ships drive the same level at once.

Everything is numpy arrays of shape (N,), so a whole population is simulated
in a single loop over time steps.
"""
import numpy as np

from . import config as C
from .level import Level, PAD_COLS

# Observation grid: every road row from the ship's row to 20 rows ahead, in
# lanes -6..+6 relative to the ship's lane (enough to see the whole 7-lane road
# from either edge). Rows are grouped into distance bands; each band reports
# the most dangerous thing in it, so a thin hole or wall can never fall
# between samples.
OBS_ROWS = 21
LOOK_LANES = np.arange(-6, 7, dtype=np.float32)
BAND_STARTS = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16])   # row offsets where bands begin
N_BANDS = len(BAND_STARTS)

KIND_CODE = np.array([0.0, 0.5, -0.5, -1.0, 1.0], np.float32)  # normal, boost, sticky, burn, fuel

OBS_MODES = ("full", "vision")


def reduce_bands(code: np.ndarray, axis: int) -> np.ndarray:
    """Collapse rows into bands. Codes: <0 means a hole, >=0 a surface height.
    A band is -1 if it has any hole, else the highest surface in it."""
    lo = np.minimum.reduceat(code, BAND_STARTS, axis=axis)
    hi = np.maximum.reduceat(code, BAND_STARTS, axis=axis)
    return np.where(lo < 0, -1.0, hi)


def reduce_kinds(kind_code: np.ndarray, axis: int) -> np.ndarray:
    """Burn beats sticky beats the best positive tile (boost/fuel)."""
    lo = np.minimum.reduceat(kind_code, BAND_STARTS, axis=axis)
    hi = np.maximum.reduceat(kind_code, BAND_STARTS, axis=axis)
    return np.where(lo < 0, lo, hi)


N_CLEAR = len(LOOK_LANES) + 2   # clear distance per lane, offset to best lane, its sign


def clear_path_features(blocked: np.ndarray) -> np.ndarray:
    """Summarise where the road is open. blocked is (..., rows, lanes), True for a
    cell the ship can't simply drive over. Returns (..., N_CLEAR):
      * for each relative lane, how far it stays clear (0 = blocked now, 1 = clear to the horizon)
      * the signed offset to the lane with the longest clear run (nearer lanes win ties)
      * the sign of that offset (-1 left, 0 stay, +1 right)
    """
    rows = blocked.shape[-2]
    first = np.where(blocked.any(-2), blocked.argmax(-2), rows)
    clear = first / rows
    best = (clear - 0.03 * np.abs(LOOK_LANES)).argmax(-1)
    offset = LOOK_LANES[best]
    return np.concatenate([clear, offset[..., None] / 6.0, np.sign(offset)[..., None]], -1)


def obs_size(mode: str) -> int:
    cells = N_BANDS * len(LOOK_LANES)
    if mode == "full":
        return 2 * cells + N_CLEAR + 9
    if mode == "vision":
        return cells + N_CLEAR + 4
    raise ValueError(mode)


class BatchSim:
    def __init__(self, level: Level, n: int, obs_mode: str = "full",
                 obs_noise: float = 0.0, rng: np.random.Generator | None = None):
        self.level = level
        self.n = n
        self.obs_mode = obs_mode
        self.obs_noise = obs_noise
        self.rng = rng or np.random.default_rng(0)
        self.rows, self.cols = level.top.shape
        self.reset()

    # ---- state ----------------------------------------------------------
    def reset(self):
        n = self.n
        self.x = np.full(n, C.LANES / 2, np.float32)
        self.y = np.zeros(n, np.float32)
        self.z = np.full(n, 2.0, np.float32)
        self.vy = np.zeros(n, np.float32)
        self.vz = np.zeros(n, np.float32)
        self.grounded = np.ones(n, bool)
        self.fuel = np.ones(n, np.float32)
        self.oxygen = np.ones(n, np.float32)
        self.reason = np.zeros(n, np.int8)        # 0 = running
        self.t = 0.0
        self.ticks = 0
        self.best_z = self.z.copy()
        self.last_progress_t = np.zeros(n, np.float32)
        self.since_jump = np.full(n, 1.0, np.float32)  # seconds since last jump press
        self.finish_t = np.zeros(n, np.float32)

    @property
    def alive(self):
        return self.reason == C.RUNNING

    # ---- grid lookups ---------------------------------------------------
    def _cell(self, z, x):
        r = np.clip(np.floor(z).astype(np.int32), 0, self.rows - 1)
        c = np.clip(np.floor(x).astype(np.int32) + PAD_COLS, 0, self.cols - 1)
        return r, c

    def top_at(self, z, x):
        r, c = self._cell(z, x)
        return self.level.top[r, c]

    def kind_at(self, z, x):
        r, c = self._cell(z, x)
        return self.level.kind[r, c]

    # ---- observations ---------------------------------------------------
    def observe(self) -> np.ndarray:
        n = self.n
        lane_c = np.floor(self.x) + 0.5
        zs = np.floor(self.z)[:, None, None] + 0.5 + np.arange(OBS_ROWS)[None, :, None]  # (N,R,1)
        xs = lane_c[:, None, None] + LOOK_LANES[None, None, :]                            # (N,1,L)
        zs, xs = np.broadcast_arrays(zs, xs)
        top = self.top_at(zs, xs)
        gap = top <= C.GAP + 1
        frac = (self.x - lane_c) * 2.0
        xn = self.x / C.LANES * 2.0 - 1.0
        vz = self.vz / C.V_MAX
        rel = top - self.y[:, None, None]
        blocked = gap | (rel > C.STEP)
        if self.obs_mode == "full":
            blocked = blocked | ((self.kind_at(zs, xs) == C.K_BURN) & ~gap)
        clear = clear_path_features(blocked)
        if self.obs_mode == "full":
            h = reduce_bands(np.where(gap, -1.0, top / C.TALL_H), axis=1)
            k = reduce_kinds(np.where(gap, 0.0, KIND_CODE[self.kind_at(zs, xs)]), axis=1)
            zfrac = self.z - np.floor(self.z)
            state = np.stack([frac, xn, self.y / C.BLOCK_H, self.vy / C.JUMP_V, vz,
                              self.grounded.astype(np.float32), self.fuel, self.oxygen, zfrac], 1)
            obs = np.concatenate([h.reshape(n, -1), k.reshape(n, -1), clear, state], 1)
        else:
            # what a screen reader can plausibly see: hole / floor / raised (relative to the ship)
            h = reduce_bands(np.where(gap, -1.0, np.where(rel > C.STEP, 1.0, 0.0)), axis=1)
            state = np.stack([frac, xn, vz, np.minimum(self.since_jump, 1.0)], 1)
            obs = np.concatenate([h.reshape(n, -1), clear, state], 1)
        obs = obs.astype(np.float32)
        if self.obs_noise > 0:
            obs += self.rng.normal(0, self.obs_noise, obs.shape).astype(np.float32)
        return obs

    # ---- physics --------------------------------------------------------
    def step(self, steer, throttle, jump):
        """steer, throttle in {-1,0,1}; jump bool. Arrays of shape (N,)."""
        dt = C.DT
        a = self.alive
        if not a.any():
            return
        steer = np.asarray(steer, np.float32)
        throttle = np.asarray(throttle, np.float32)
        jump = np.asarray(jump, bool)

        x, y, z, vy, vz = self.x.copy(), self.y.copy(), self.z.copy(), self.vy.copy(), self.vz.copy()
        grounded = self.grounded.copy()
        reason = self.reason.copy()

        # speed
        has_fuel = self.fuel > 0
        acc = np.where(throttle > 0, C.ACCEL * has_fuel, np.where(throttle < 0, -C.BRAKE, 0.0))
        acc = np.where(~has_fuel, -C.BRAKE * 0.3, acc)
        vz = np.clip(vz + acc * dt, 0.0, C.V_MAX)

        # jump + gravity
        start = grounded & jump
        vy = np.where(start, C.JUMP_V, vy)
        grounded &= ~start
        vy = vy - C.GRAVITY * dt
        self.since_jump = np.where(jump, 0.0, self.since_jump + dt)

        # forward motion with wall check
        nz = z + vz * dt
        front = nz + C.HALF_L
        xl, xr = x - C.HALF_W + 1e-3, x + C.HALF_W - 1e-3
        top_front = np.maximum(self.top_at(front, xl), self.top_at(front, xr))
        blocked = (top_front > y + C.STEP) & (np.floor(front) > np.floor(z + C.HALF_L))
        crash = blocked & (vz > C.CRASH_SPEED)
        reason = np.where(crash & a, C.CRASHED, reason)
        nz = np.where(blocked, np.floor(front) - C.HALF_L - 1e-3, nz)
        vz = np.where(blocked, 0.0, vz)
        z = nz

        # lateral motion with wall check
        nx = x + steer * C.STEER_SPEED * dt
        zb, zf = z - C.HALF_L + 1e-3, z + C.HALF_L - 1e-3
        edge = np.where(steer > 0, nx + C.HALF_W, nx - C.HALF_W)
        top_side = np.maximum(self.top_at(zb, edge), self.top_at(zf, edge))
        sblocked = (steer != 0) & (top_side > y + C.STEP) & (np.floor(edge) != np.floor(
            np.where(steer > 0, x + C.HALF_W, x - C.HALF_W)))
        nx = np.where(sblocked & (steer > 0), np.floor(edge) - C.HALF_W - 1e-3, nx)
        nx = np.where(sblocked & (steer < 0), np.floor(edge) + 1 + C.HALF_W + 1e-3, nx)
        x = nx

        # vertical motion and landing
        xl, xr = x - C.HALF_W + 1e-3, x + C.HALF_W - 1e-3
        support = np.maximum.reduce([self.top_at(zb, xl), self.top_at(zb, xr),
                                     self.top_at(zf, xl), self.top_at(zf, xr)])
        ny = y + vy * dt
        land = (ny <= support) & (y >= support - C.STEP)
        y = np.where(land, support, ny)
        vy = np.where(land, 0.0, vy)
        grounded = land

        # tile effects (centre tile, only if it is the one we stand on)
        ctop = self.top_at(z, x)
        on_c = grounded & (np.abs(ctop - y) < 1e-4)
        k = np.where(on_c, self.kind_at(z, x), C.K_NORMAL)
        reason = np.where(a & (reason == 0) & (k == C.K_BURN), C.BURNED, reason)
        fuel = np.where(k == C.K_FUEL, 1.0, self.fuel)
        vz = np.where(k == C.K_BOOST, C.V_MAX, vz)
        vz = np.where(k == C.K_STICKY, vz * 0.88, vz)

        # resources
        fuel = np.maximum(fuel - vz * dt / self.level.fuel_rows, 0.0)
        oxygen = np.maximum(self.oxygen - dt / self.level.oxygen_secs, 0.0)

        t = self.t + dt
        progressed = z > self.best_z + 0.05
        best_z = np.where(progressed, z, self.best_z)
        last_p = np.where(progressed, t, self.last_progress_t)

        ok = a & (reason == 0)
        reason = np.where(ok & (y < C.FALL_DEATH_Y), C.FELL, reason)
        ok = a & (reason == 0)
        reason = np.where(ok & (z >= self.level.length), C.WIN, reason)
        ok = a & (reason == 0)
        reason = np.where(ok & (oxygen <= 0), C.NO_OXYGEN, reason)
        reason = np.where(ok & (fuel <= 0) & (vz <= 0), C.NO_FUEL, reason)
        ok = a & (reason == 0)
        reason = np.where(ok & (t - last_p > C.STALL_SECONDS), C.STALLED, reason)
        ok = a & (reason == 0)
        reason = np.where(ok & (t > self.level.max_seconds), C.TIMEOUT, reason)
        self.finish_t = np.where(a & (reason == C.WIN), t, self.finish_t)

        # commit only for ships that were alive at the start of the tick
        for name, val in (("x", x), ("y", y), ("z", z), ("vy", vy), ("vz", vz),
                          ("grounded", grounded), ("fuel", fuel), ("oxygen", oxygen),
                          ("best_z", best_z), ("last_progress_t", last_p)):
            old = getattr(self, name)
            setattr(self, name, np.where(a, val, old).astype(old.dtype))
        self.reason = np.where(a, reason, self.reason).astype(np.int8)
        self.t = t
        self.ticks += 1

    # ---- scoring --------------------------------------------------------
    def fitness(self) -> np.ndarray:
        """The goal is only to reach the end: progress along the road, plus a bonus
        for finishing. There is no reward for speed; oxygen supplies the time limit."""
        progress = np.clip(self.best_z / self.level.length, 0, 1)
        win = self.reason == C.WIN
        return (1000 * progress + 1000 * win).astype(np.float32)
