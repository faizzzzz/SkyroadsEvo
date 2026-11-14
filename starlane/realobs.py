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
# The port's own cell types (game_play.c): 1 tunnel, 2 wall, 3 arch, 4 double wall, 5 double arch.
# Arches can be driven under like tunnels; wall and arch roofs are 2560 above the road, double ones
# 5120. CELL_TYPES_V2 = True reads them that way; False keeps the original reading (every object but a
# tunnel is a block, and low blocks only clear above 2600, so a ship riding on a roof saw a wall).
CELL_TYPES_V2 = False
REACH_V2 = False               # the correct reading for the 15 jump-aware inputs only
CELL_ARCH, CELL_DOUBLE_WALL, CELL_DOUBLE_ARCH = 3, 4, 5
ROOF, DOUBLE_ROOF = 2560, 5120
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


def build_obs(raw: np.ndarray, x: float, y: float, speed: float, gravity: float,
              fuel: float = 1.0, oxygen: float = 1.0) -> list:
    """raw: (OBS_ROWS, 7) uint16 cells, starting at the ship's current row.
    x, y, speed in the game's units; fuel and oxygen as shares of full. Returns the 222-number input
    list: road grid (143), clear path (15), ship state (5), road end (2), the jump-aware clear path
    (15), the jump-aware clear path with the cells read by the game's rules (15), special tiles ahead
    (20), what the ship stands on (5), fuel and oxygen (2).

    The road-derived parts depend only on the cells in view, the ship's lane and whether it is above
    the low blocks, which stay the same for several frames in a row, so they are cached (building them
    was about three quarters of a flight's run time)."""
    lane_x = lane_position(x)
    lane = int(np.clip(np.floor(lane_x), 0, C.LANES - 1))
    # level2 (correct cell types): 0 road level, 1 on or above a wall roof, 2 on or above a double roof
    level2 = 2 if y >= GROUND + DOUBLE_ROOF - 40 else (1 if y >= GROUND + ROOF - 40 else 0)
    level = level2 if CELL_TYPES_V2 else (1 if y > GROUND + LOW_BLOCK_CLEAR else 0)
    key = (raw.tobytes(), raw.shape[0], lane, level, level2, CELL_TYPES_V2, REACH_V2)
    road = _ROAD_CACHE.get(key)
    if road is None:
        if len(_ROAD_CACHE) >= _ROAD_CACHE_MAX:
            _ROAD_CACHE.clear()
        road = _ROAD_CACHE[key] = _road_inputs(raw, lane, level, level2)
    centre = lane + 0.5
    state = [(lane_x - centre) * 2, lane_x / C.LANES * 2 - 1, min(1.0, speed / SPEED_FULL),
             float(np.clip((y - GROUND) / 6000.0, -1.0, 1.0)), gravity]
    end = road_end_inputs(raw, lane_x)
    return (road[0] + road[1] + np.array(state, np.float32).tolist()
            + np.array(end, np.float32).tolist() + road[2] + road[3] + road[4]
            + standing_on(raw[0, lane], y) + [float(np.float32(fuel)), float(np.float32(oxygen))])


def standing_on(cell: int, y: float) -> list:
    """inputs 216-220, what the ship is on right now: road, a block roof, a tunnel roof (curved: only
    the middle holds the ship; the lane-position input says how far off it is), and whether that
    surface is slippery or sticky (brake). All 0 in the air."""
    cell = int(cell)
    hi, floor, roof = cell >> 8, cell & 0xF, (cell >> 4) & 0xF
    h = y - GROUND
    road = roof_top = tunnel_top = 0.0
    tile = 0
    if abs(h) < 40 and (hi in (0, CELL_TUNNEL, CELL_ARCH, CELL_DOUBLE_ARCH)) and floor:
        road, tile = 1.0, floor
    elif hi == CELL_TUNNEL and 40 <= h <= ROOF + 40:
        tunnel_top = 1.0
    elif (hi in (CELL_LOW_BLOCK, CELL_ARCH) and abs(h - ROOF) < 40) or \
            (hi in (CELL_DOUBLE_WALL, CELL_DOUBLE_ARCH) and abs(h - DOUBLE_ROOF) < 40):
        roof_top, tile = 1.0, roof
    return [road, roof_top, tunnel_top, float(tile == TILE_SLIPPERY), float(tile == TILE_BRAKE)]


def _cell_grid(lo, hi, level, v2):
    """Cells as the pilot sees them: hole -1, road 0, tunnel (or arch, v2) 0.5, block 1."""
    grid = np.full(lo.shape, -1.0, np.float32)            # empty cell: a hole
    grid[(hi == 0) & (lo > 0)] = 0.0                       # plain road
    grid[hi == CELL_TUNNEL] = TUNNEL_CODE                  # tunnels can be driven through
    grid[(hi != 0) & (hi != CELL_TUNNEL)] = 1.0            # blocks
    grid[(hi == 0) & (lo == TILE_DEADLY)] = 1.0            # deadly tiles are as bad as walls
    if v2:
        grid[(hi == CELL_ARCH) | (hi == CELL_DOUBLE_ARCH)] = TUNNEL_CODE   # arches: drive underneath
        if level >= 1:                                     # on a roof: walls and arches are floor
            grid[(hi == CELL_LOW_BLOCK) | (hi == CELL_ARCH)] = 0.0
            grid[(hi == 0) & (lo == TILE_DEADLY)] = 0.0
        if level >= 2:
            grid[(hi == CELL_DOUBLE_WALL) | (hi == CELL_DOUBLE_ARCH)] = 0.0
    elif level:
        grid[hi == CELL_LOW_BLOCK] = 0.0
        grid[(hi == 0) & (lo == TILE_DEADLY)] = 0.0        # ...unless flying over them
    return grid


TILE_BRAKE, TILE_SLIPPERY, TILE_SUPPLIES, TILE_SPEED, TILE_FIRE = 2, 8, 9, 10, 12
SPECIAL_TILES = (TILE_SPEED, TILE_BRAKE, TILE_SLIPPERY, TILE_SUPPLIES)
TILE_LANES = (-2, -1, 0, 1, 2)
GAP_ROWS_V3 = 2                # at top speed a jump clears 2-row holes easily, 3 only from the last 0.3 row


def _surface_tiles(lo, hi, level):
    """The tile the ship would be on in each cell at this height (0 = nothing to stand on): the floor
    for plain cells, tunnels and arch openings at road level, the roof tile on block roofs."""
    floor, roof = lo & 0xF, (lo >> 4) & 0xF
    tile = np.zeros(lo.shape, np.int64)
    if level == 0:
        under = (hi == 0) | (hi == CELL_TUNNEL) | (hi == CELL_ARCH) | (hi == CELL_DOUBLE_ARCH)
        tile[under] = floor[under]
    else:
        tops = (hi == CELL_LOW_BLOCK) | (hi == CELL_ARCH)
        if level >= 2:
            tops |= (hi == CELL_DOUBLE_WALL) | (hi == CELL_DOUBLE_ARCH)
        tile[tops] = roof[tops]
    return tile


def _cell_grid_v3(lo, hi, level):
    """Cells read with the game's own rules (game_play.c, checked on test strips): hole -1, floor 0,
    passage 0.5 (tunnel or arch with a floor), blocked 1. Tunnels and arches over a hole are holes at
    road level; fire under them or on a roof is blocked; tall blocks are blocked below their roof."""
    floor, roof = lo & 0xF, (lo >> 4) & 0xF
    grid = np.full(lo.shape, -1.0, np.float32)
    plain = hi == 0
    grid[plain & (floor > 0)] = 0.0
    grid[plain & (floor == TILE_FIRE)] = 1.0 if level == 0 else 0.0     # flying over fire is fine
    passage = (hi == CELL_TUNNEL) | (hi == CELL_ARCH) | (hi == CELL_DOUBLE_ARCH)
    wall = (hi == CELL_LOW_BLOCK) | (hi == CELL_DOUBLE_WALL)
    if level == 0:
        grid[passage & (floor > 0)] = TUNNEL_CODE
        grid[passage & (floor == TILE_FIRE)] = 1.0
        grid[wall] = 1.0
    else:
        grid[hi == CELL_TUNNEL] = 0.0                       # a (curved) tunnel roof
        low = (hi == CELL_LOW_BLOCK) | (hi == CELL_ARCH)
        tall = (hi == CELL_DOUBLE_WALL) | (hi == CELL_DOUBLE_ARCH)
        grid[low] = np.where(roof[low] == TILE_FIRE, 1.0, 0.0)
        grid[tall] = 1.0 if level == 1 else np.where(roof[tall] == TILE_FIRE, 1.0, 0.0)
    return grid


def _road_inputs(raw: np.ndarray, lane: int, level: int, level2: int) -> tuple:
    """The parts of build_obs that depend only on the road: (grid bands, clear path, jump-aware
    clear path), each a list of float32 values. level / level2: see build_obs."""
    lo, hi = raw & 255, raw >> 8
    lanes = lane + LOOK_LANES.astype(int)
    inside = (lanes >= 0) & (lanes < C.LANES)

    def view(grid):
        code = np.full((OBS_ROWS, len(LOOK_LANES)), -1.0, np.float32)
        code[:, inside] = grid[:, lanes[inside]]
        return code

    code = view(_cell_grid(lo, hi, level, CELL_TYPES_V2))
    blocked = (code != 0) & (code != TUNNEL_CODE)
    # the jump-aware clear path comes as 15 extra inputs at the end, so the original 165 keep their
    # meaning: replacing them cut Road 1 finishes from 43 to 16 and Road 2 from 24 to 1 in 150 pilots.
    # With REACH_V2 it also reads the cell types correctly (arches, roofs), on its own.
    def reach_of(v2):
        code_r = view(_cell_grid(lo, hi, level2 if v2 else level, v2))
        reach = (code_r != 0) & (code_r != TUNNEL_CODE)
        if LOW_BLOCK_PLATFORM:
            roofs = (hi == CELL_LOW_BLOCK) | ((hi == CELL_ARCH) if v2 else False)
            low = np.zeros(code.shape, bool)
            low[:, inside] = roofs[:, lanes[inside]]
            reach &= ~low                                   # a platform to jump onto, not a wall
        return jump_aware(reach, code_r == -1.0)

    # inputs 181-195: the jump-aware clear path with the cells read by the game's own rules (arches
    # driven under, roofs ridden on, tunnels and arches over holes are holes, fire on roofs, 2-row
    # gaps). Added alongside: correcting the existing inputs in place left none of the 13 pilots that
    # finished Roads 1-4 still finishing them.
    code3 = view(_cell_grid_v3(lo, hi, level2))
    reach3 = (code3 != 0) & (code3 != TUNNEL_CODE)
    platforms = ((hi == CELL_LOW_BLOCK) | (hi == CELL_ARCH)) & (((lo >> 4) & 0xF) != TILE_FIRE)
    low = np.zeros(code3.shape, bool)
    low[:, inside] = platforms[:, lanes[inside]]
    reach3 = jump_aware(reach3 & ~low, code3 == -1.0, GAP_ROWS_V3)

    # inputs 196-215: for the ship's lane and two either side, how near the next speed, brake,
    # slippery and supplies tile is (1 = here, 0 = none in view)
    tiles = _surface_tiles(lo, hi, level2)
    near = []
    for rel in TILE_LANES:
        col = lane + rel
        for t in SPECIAL_TILES:
            if 0 <= col < C.LANES and (tiles[:, col] == t).any():
                near.append(1.0 - int(np.argmax(tiles[:, col] == t)) / OBS_ROWS)
            else:
                near.append(0.0)

    f32 = lambda a: np.asarray(a, np.float32).ravel().tolist()
    return (f32(reduce_bands(code, axis=0)), f32(clear_path_features(blocked)),
            f32(clear_path_features(reach_of(CELL_TYPES_V2 or REACH_V2))), f32(clear_path_features(reach3)),
            f32(near))
