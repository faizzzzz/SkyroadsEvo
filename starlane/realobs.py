"""The pilot's inputs, built from a grid of road cells in the game's own encoding.

Both the live game bridge (cells read from the game's memory) and the matched simulator
(cells from its own generated roads) call build_obs, so a pilot sees exactly the same kind
of information in both places.

Cell encoding (16 bits): low byte is the surface (0 = nothing, 0x0C = deadly, other = road),
high byte is an object (0 none, 1 tunnel, 2/3 low block, 4 tall block).
"""
import numpy as np

from . import config as C
from .sim import LOOK_LANES, OBS_ROWS, clear_path_features, reduce_bands

GROUND = 10240                 # height at road level
# The game's own lane geometry (from the skyroads-sdl port's game logic, which reproduces DOSBox):
# the road's left edge is 95 pixels and each lane 46 pixels, at 128 lateral units per pixel. An
# earlier mapping measured by eye from DOSBox (18432 + 4096 per lane, shifted 0.435 lane) put the
# ship up to a lane too far right on the right half of the road.
X_LEFT, X_LANE = 95 * 128, 46 * 128
SPEED_FULL = 8704
CELL_TUNNEL = 1
CELL_LOW_BLOCK = 2
TILE_DEADLY = 0x0C
LOW_BLOCK_CLEAR = 2600         # height above road level that clears a low block or tunnel top
TUNNEL_CODE = 0.5              # a tunnel in the grid: not floor, not a block
REFERENCE_JUMP = 5195          # peak jump height above road level at the standard gravity
SHIP_HALF_WIDTH = 14.5 / 46    # the ship is 29 pixels wide
# Clear-path inputs treat a hole this many rows long or shorter as passable when road follows it,
# since a jump covers it (about 3 rows at top speed and standard gravity). Before this, a hole counted
# as blocked at once, so at Road 3's row 43 the lanes the road continues on (holes for 3 rows, then
# road) read as blocked and the dead-end lane as the longest clear path: the inputs pointed pilots
# away from the only way on. 0 restores the old behaviour.
JUMP_GAP_ROWS = 3
# Clear-path inputs treat low blocks as part of the way on (the ship can jump onto them and ride on top),
# like tunnels. Road 3 continues past row 49 only along a strip of low blocks on the far right.
LOW_BLOCK_PLATFORM = True


def lane_position(x: float) -> float:
    """Lateral position in lanes, such that lane n covers [n, n + 1) (the game's own lanes)."""
    return (x - X_LEFT) / X_LANE


def lane_of(x: float) -> int:
    return int(np.floor(lane_position(x)))


class Steering:
    """Proportional steering on an on/off key, like a stick on a keyboard.

    The size of the pilot's steering output sets how often the key is held: a strong output holds
    it every update (as before), a gentle one taps it now and then, for small corrections on strips
    barely wider than the ship. One instance per flight.
    """
    DEADBAND = 0.15

    def __init__(self):
        self.acc = 0.0

    def __call__(self, out: float) -> int:
        duty = min(1.0, max(0.0, (abs(out) - self.DEADBAND) / (1.0 - self.DEADBAND)))
        self.acc += duty
        if self.acc >= 1.0:
            self.acc -= 1.0
            return 1 if out > 0 else -1
        return 0


class JumpTap:
    """The jump output as a deliberate press, not a held button.

    In the game, holding the jump key re-jumps the moment the ship lands, so a pilot whose output
    simply stays on bounces down the whole road without looking. Here the output switching on presses
    the key until that jump has started (a press in the air waits for the landing, as a player taps
    slightly early); to jump again the output must switch off first. Doing nothing means riding on.
    One instance per flight.
    """

    def __init__(self):
        self.armed, self.pressing, self.settled = True, False, False

    def __call__(self, out: float, rising: bool) -> bool:
        """rising: the ship is going up (a jump under way)."""
        if out <= 0.5:
            self.armed, self.pressing = True, False
        elif self.armed:
            self.armed, self.pressing, self.settled = False, True, False
        if self.pressing:
            if not rising:
                self.settled = True           # on the ground or coming down: the next rise is ours
            elif self.settled:
                self.pressing = False         # our jump is under way: let go
        return self.pressing


def gravity_input(jump_peak_above_road: float) -> float:
    factor = REFERENCE_JUMP / max(jump_peak_above_road, 1.0)
    return float(np.clip((factor - 1.0) * 5.0, -1.0, 1.0))


END_GAP_ROWS = 4               # empty rows after the last cell in view: that's the road's end


def road_end_inputs(raw: np.ndarray, lane_x: float):
    """Two inputs about the road's end, once it is in view (as a player sees it on screen): how close
    it is (0 = not in view, rising to 1 at the end) and how far the final tunnel is to the right
    (+) or left (-), in units of two lanes. Every road ends in a tunnel that must be driven into."""
    used = np.nonzero((raw != 0).any(axis=1))[0]
    if not len(used) or used[-1] > len(raw) - 1 - END_GAP_ROWS:
        return 0.0, 0.0
    last = int(used[-1])
    tunnel = np.nonzero((raw[last] >> 8) == CELL_TUNNEL)[0]
    lanes = tunnel if len(tunnel) else np.nonzero(raw[last])[0]
    target = float(lanes[np.argmin(np.abs(lanes + 0.5 - lane_x))]) + 0.5
    return 1.0 - last / len(raw), float(np.clip((target - lane_x) / 2.0, -1.0, 1.0))


def jump_aware(blocked: np.ndarray, hole: np.ndarray, gap: int = None) -> np.ndarray:
    """blocked, hole: (rows, lanes). Clears hole runs of at most `gap` rows that end on a cell the
    ship can drive on, so the clear-path inputs count a jumpable gap as part of the way on."""
    gap = JUMP_GAP_ROWS if gap is None else gap
    if gap <= 0:
        return blocked
    rows = hole.shape[0]
    idx = np.arange(rows)[:, None]
    # for every cell, the first non-hole row at or after it and the last one at or before it
    after = np.minimum.accumulate(np.where(hole, rows, idx)[::-1], axis=0)[::-1]
    before = np.maximum.accumulate(np.where(hole, -1, idx), axis=0)
    landing = np.take_along_axis(blocked, np.minimum(after, rows - 1), axis=0)
    jumpable = hole & (after - before - 1 <= gap) & (after < rows) & ~landing
    return blocked & ~jumpable


_ROAD_CACHE = {}
_ROAD_CACHE_MAX = 200_000


def build_obs(raw: np.ndarray, x: float, y: float, speed: float, gravity: float) -> list:
    """raw: (OBS_ROWS, 7) uint16 cells, starting at the ship's current row.
    x, y, speed in the game's units. Returns the 180-number input list: road grid (143), clear path (15),
    ship state (5), road end (2), and the jump-aware clear path (15).

    The road-derived parts depend only on the cells in view, the ship's lane and whether it is above
    the low blocks, which stay the same for several frames in a row, so they are cached (building them
    was about three quarters of a flight's run time)."""
    lane_x = lane_position(x)
    lane = int(np.clip(np.floor(lane_x), 0, C.LANES - 1))
    high = y > GROUND + LOW_BLOCK_CLEAR
    key = (raw.tobytes(), raw.shape[0], lane, high)
    road = _ROAD_CACHE.get(key)
    if road is None:
        if len(_ROAD_CACHE) >= _ROAD_CACHE_MAX:
            _ROAD_CACHE.clear()
        road = _ROAD_CACHE[key] = _road_inputs(raw, lane, high)
    centre = lane + 0.5
    state = [(lane_x - centre) * 2, lane_x / C.LANES * 2 - 1, min(1.0, speed / SPEED_FULL),
             float(np.clip((y - GROUND) / 6000.0, -1.0, 1.0)), gravity]
    end = road_end_inputs(raw, lane_x)
    return (road[0] + road[1] + np.array(state, np.float32).tolist()
            + np.array(end, np.float32).tolist() + road[2])


def _road_inputs(raw: np.ndarray, lane: int, high: bool) -> tuple:
    """The parts of build_obs that depend only on the road: (grid bands, clear path, jump-aware
    clear path), each a list of float32 values."""
    lo, hi = raw & 255, raw >> 8
    grid = np.full(raw.shape, -1.0, np.float32)            # empty cell: a hole
    grid[(hi == 0) & (lo > 0)] = 0.0                       # plain road
    grid[hi == CELL_TUNNEL] = TUNNEL_CODE                  # tunnels can be driven through
    grid[(hi != 0) & (hi != CELL_TUNNEL)] = 1.0            # blocks
    grid[(hi == 0) & (lo == TILE_DEADLY)] = 1.0            # deadly tiles are as bad as walls
    if high:
        grid[hi == CELL_LOW_BLOCK] = 0.0
        grid[(hi == 0) & (lo == TILE_DEADLY)] = 0.0        # ...unless flying over them
    code = np.full((OBS_ROWS, len(LOOK_LANES)), -1.0, np.float32)
    lanes = lane + LOOK_LANES.astype(int)
    inside = (lanes >= 0) & (lanes < C.LANES)
    code[:, inside] = grid[:, lanes[inside]]
    blocked = (code != 0) & (code != TUNNEL_CODE)
    # the jump-aware clear path comes as 15 extra inputs at the end, so the original 165 keep their
    # meaning: replacing them cut Road 1 finishes from 43 to 16 and Road 2 from 24 to 1 in 150 pilots
    reach = blocked.copy()
    if LOW_BLOCK_PLATFORM:
        low = np.zeros(code.shape, bool)
        low[:, inside] = (hi == CELL_LOW_BLOCK)[:, lanes[inside]]
        reach &= ~low                                       # a platform to jump onto, not a wall
    reach = jump_aware(reach, code == -1.0)
    f32 = lambda a: np.asarray(a, np.float32).ravel().tolist()
    return f32(reduce_bands(code, axis=0)), f32(clear_path_features(blocked)), f32(clear_path_features(reach))
