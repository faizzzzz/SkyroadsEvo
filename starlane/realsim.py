"""A simulator matched to the real game's measured physics, in the game's own units.

Measured from the running game (see the README):
  the game updates the ship about 26 times a second, and reacts about one update after a key;
  acceleration 2156 speed units/s, braking 2030, no drag, top speed 8704 = 6.03 rows/s;
  steering about 3300 + 3.15 * speed lateral units/s (one lane = 4096), levelling off near 23500;
  a jump peaks 5195 above road level (gravity about 148000 units/s^2 on most roads; one road
  in ten measured lower, peak 6072);
  lane boundaries sit 0.435 lane left of the nominal grid, and the ship (0.195 lane either side
  of its centre) stays up while any part of it is over road;
  road level 10240, tunnel tops 2560 above it.
Still estimated, not measured: block heights, what counts as a crash against a block, fuel and
oxygen (not modelled), and the effect of surface codes other than deadly tiles.
Pilots see the same inputs as on the real game (starlane/realobs.py).
"""
import numpy as np

from .realobs import (GROUND, SHIP_HALF_WIDTH, SPEED_FULL, TILE_DEADLY, Steering, build_obs,
                      gravity_input, lane_position)
from .realroads import LANES, START_ROW, Road
from .sim import OBS_ROWS

DT = 1.0 / 26.0                         # the game's update rate
ACCEL, BRAKE = 2156.0, 2030.0
ROWS_PER_SPEED = 6.03 / SPEED_FULL
STEER_BASE, STEER_PER_SPEED, STEER_MAX = 3300.0, 3.15, 23500.0
JUMP_V = 42065.0                        # calibrated (at 26 updates/s) so a jump peaks 5195 above the road
TOP_LOW = GROUND + 2560
TOP_TALL = GROUND + 8000
TUNNEL_INSIDE = GROUND + 300            # inside a tunnel the ship can't be higher than this
TOL = 60.0
APPROACH_ROWS = 5
RUNNING, WIN, FELL, CRASHED, BURNED, STALLED, TIMEOUT = range(7)
REASONS = ["running", "finished", "fell", "crashed", "burned", "stalled", "timeout"]


def _top(cell):
    """Height of the highest surface in a cell (None if nothing), and whether it has road under it."""
    hi, lo = cell >> 8, cell & 255
    floor = lo > 0
    if hi in (2, 3, 5):
        return TOP_LOW, floor
    if hi == 4:
        return TOP_TALL, floor
    if hi == 1:
        return TOP_LOW, floor                # a tunnel's top is a surface too
    return (GROUND if floor else None), floor


class Ship:
    __slots__ = ("z", "x", "y", "vy", "speed", "grounded", "reason", "progress", "best_z", "last_move",
                 "t", "approach_ticks", "approach_good", "jump_ticks", "ticks", "pending")

    def __init__(self):
        self.z, self.x, self.y, self.vy, self.speed = float(START_ROW), 32768.0, float(GROUND), 0.0, 0.0
        self.grounded, self.reason, self.progress, self.best_z = True, RUNNING, 0.0, float(START_ROW)
        self.last_move = self.t = 0.0
        self.approach_ticks = self.approach_good = self.jump_ticks = self.ticks = 0
        self.pending = []                       # keys take effect one (sometimes two) updates later


class Physics:
    """The measured constants, each nudged a little per road so pilots can't depend on exact timing.
    Every pilot on the same road gets the same nudges (they come from the road's seed)."""

    def __init__(self, seed=None, vary=0.0):
        rng = np.random.default_rng(None if seed is None else seed + 7_919_311)

        def f(spread):
            return 1.0 + (rng.uniform(-spread, spread) if vary else 0.0)
        self.accel, self.brake = ACCEL * f(0.08 * vary), BRAKE * f(0.08 * vary)
        steer = f(0.08 * vary)
        self.steer_base, self.steer_per, self.steer_max = STEER_BASE * steer, STEER_PER_SPEED * steer, STEER_MAX * steer
        self.jump_v = JUMP_V * f(0.03 * vary)
        self.gravity_scale = f(0.04 * vary)
        self.delay = 2 if vary and rng.random() < 0.2 * vary else 1   # keys sometimes land an update later


class RoadSim:
    def __init__(self, road: Road, vary: float = 0.0):
        self.road = road
        self.cells = road.cells
        self.p = Physics(road.seed, vary)
        self.gravity = road.gravity * self.p.gravity_scale
        # the pilot's gravity input comes from the jump height it would measure on this road
        self.grav_in = gravity_input(self.p.jump_v ** 2 / (2 * self.gravity))

    def cell(self, row, lane):
        if lane < 0 or lane >= LANES or row < 0 or row >= len(self.cells):
            return 0
        return int(self.cells[row, lane])

    def lanes_under(self, x):
        """Lanes the ship's body overlaps."""
        lx = lane_position(x)
        return range(int(np.floor(lx - SHIP_HALF_WIDTH)), int(np.floor(lx + SHIP_HALF_WIDTH)) + 1)

    def observe(self, s: Ship):
        row = int(s.z)
        raw = np.zeros((OBS_ROWS, LANES), np.uint16)
        part = self.cells[row:row + OBS_ROWS]
        raw[:len(part)] = part
        return build_obs(raw, s.x, s.y, s.speed, self.grav_in)

    def step(self, s: Ship, steer: int, throttle: int, jump: bool):
        if s.reason != RUNNING:
            return
        road, L = self.road, self.road.length
        p = self.p
        s.pending.append((steer, throttle, jump))
        steer, throttle, jump = s.pending.pop(0) if len(s.pending) > p.delay else (0, 0, False)
        s.ticks += 1
        s.t += DT
        if throttle > 0:
            s.speed = min(SPEED_FULL, s.speed + p.accel * DT)
        elif throttle < 0:
            s.speed = max(0.0, s.speed - p.brake * DT)
        if jump:
            s.jump_ticks += 1
            if s.grounded:
                s.vy, s.grounded = p.jump_v, False
        # forward, against whatever the ship's body meets in the next row
        nz = s.z + s.speed * ROWS_PER_SPEED * DT
        if int(nz) > int(s.z):
            for lane in self.lanes_under(s.x):
                c = self.cell(int(nz), lane)
                top, floor = _top(c)
                hi = c >> 8
                if hi in (2, 3, 4, 5) and s.y < top - TOL:
                    s.reason = CRASHED
                    return
                if hi == 1 and TUNNEL_INSIDE < s.y < TOP_LOW - TOL:
                    s.reason = CRASHED                 # hit the tunnel's face
                    return
                if floor and s.y < GROUND - TOL:
                    s.reason = CRASHED                 # dipped below road level into the next row's edge
                    return
        s.z = nz
        # sideways, unless the body would push into something taller than the ship
        rate = min(p.steer_max, p.steer_base + p.steer_per * s.speed)
        nx = s.x + steer * rate * DT
        if steer:
            for lane in self.lanes_under(nx):
                if lane in self.lanes_under(s.x):
                    continue
                top, _ = _top(self.cell(int(s.z), lane))
                if top is not None and top > s.y + TOL:
                    nx = s.x
                    break
        s.x = nx
        # what is under the body
        row = int(s.z)
        support, in_tunnel = None, False
        centre_cell = self.cell(row, int(np.floor(lane_position(s.x))))
        for lane in self.lanes_under(s.x):
            c = self.cell(row, lane)
            top, floor = _top(c)
            hi = c >> 8
            if hi == 1 and floor and s.y <= TUNNEL_INSIDE:
                in_tunnel = True
                support = GROUND if support is None else max(support, GROUND)
            elif top is not None and s.y >= top - TOL:
                support = top if support is None else max(support, top)
            elif floor and hi in (0, 1) and s.y >= GROUND - TOL:
                support = GROUND if support is None else max(support, GROUND)
        if not s.grounded:
            s.vy -= self.gravity * DT
        ny = s.y + s.vy * DT
        if in_tunnel:
            ny = min(ny, TUNNEL_INSIDE)               # the tunnel's ceiling
        if support is not None and ny <= support:
            s.y, s.vy, s.grounded = support, 0.0, True
        else:
            s.y, s.grounded = ny, False
            if support is None and s.vy == 0:
                s.vy = -1.0
        # outcomes
        if row >= L - 1 and in_tunnel:
            s.progress, s.reason = 1.0, WIN
            return
        if s.y <= 0:
            s.reason = FELL
            return
        if s.grounded and abs(s.y - GROUND) < 1 and (centre_cell & 255) == TILE_DEADLY and centre_cell >> 8 == 0:
            s.reason = BURNED
            return
        if L - 1 - APPROACH_ROWS <= s.z < L - 1:
            s.approach_ticks += 1
            s.approach_good += (s.grounded and abs(s.y - GROUND) < 1
                                and any(l in road.tunnel_lanes for l in self.lanes_under(s.x)))
        if s.z > s.best_z + 1e-3:
            s.best_z, s.last_move = s.z, s.t
            if s.y >= GROUND - 300:                    # rows covered while falling don't count (as in live.fly)
                s.progress = min(1.0, (s.z - START_ROW) / max(1, L - START_ROW))
        moved = s.best_z > START_ROW + 0.05
        if s.t - s.last_move > (1.5 if moved else 5.0):
            s.reason = STALLED
        elif s.t > 150:
            s.reason = TIMEOUT

    def result(self, s: Ship):
        end = s.approach_good / s.approach_ticks if s.approach_ticks else 0.0
        won = s.reason == WIN
        return {"progress": 1.0 if won else s.progress, "won": won, "end_score": 1.0 if won else end,
                "why": REASONS[s.reason], "jump_frac": s.jump_ticks / max(1, s.ticks)}


def to_keys(out, steering):
    thr = 1 if out[1] > 0.33 else (-1 if out[1] < -0.33 else 0)
    return steering(out[0]), thr, out[2] > 0.5


def fly(nets, road: Road, vary: float = 0.0):
    """Fly several networks on one road, side by side. Returns a list of result dicts.
    vary: how much to nudge the physics on this road (0 = exactly as measured, 1 = full spread)."""
    sim = RoadSim(road, vary)
    ships = [Ship() for _ in nets]
    steering = [Steering() for _ in nets]
    alive = list(range(len(ships)))
    while alive:
        for i in alive:
            s = ships[i]
            sim.step(s, *to_keys(nets[i].activate(sim.observe(s)), steering[i]))
        alive = [i for i in alive if ships[i].reason == RUNNING]
    return [sim.result(s) for s in ships]
