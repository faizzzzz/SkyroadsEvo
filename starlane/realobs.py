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


def build_obs(raw: np.ndarray, x: float, y: float, speed: float, gravity: float) -> list:
    """raw: (OBS_ROWS, 7) uint16 cells, starting at the ship's current row.
    x, y, speed in the game's units. Returns the 165-number input list (the last two: road_end_inputs)."""
    lane_x = lane_position(x)
    lane = int(np.clip(np.floor(lane_x), 0, C.LANES - 1))
    lo, hi = raw & 255, raw >> 8
    grid = np.full(raw.shape, -1.0, np.float32)            # empty cell: a hole
    grid[(hi == 0) & (lo > 0)] = 0.0                       # plain road
    grid[hi == CELL_TUNNEL] = TUNNEL_CODE                  # tunnels can be driven through
    grid[(hi != 0) & (hi != CELL_TUNNEL)] = 1.0            # blocks
    grid[(hi == 0) & (lo == TILE_DEADLY)] = 1.0            # deadly tiles are as bad as walls
    if y > GROUND + LOW_BLOCK_CLEAR:
        grid[hi == CELL_LOW_BLOCK] = 0.0
        grid[(hi == 0) & (lo == TILE_DEADLY)] = 0.0        # ...unless flying over them
    code = np.full((OBS_ROWS, len(LOOK_LANES)), -1.0, np.float32)
    lanes = lane + LOOK_LANES.astype(int)
    inside = (lanes >= 0) & (lanes < C.LANES)
    code[:, inside] = grid[:, lanes[inside]]
    centre = lane + 0.5
    state = np.array([(lane_x - centre) * 2, lane_x / C.LANES * 2 - 1,
                      min(1.0, speed / SPEED_FULL),
                      float(np.clip((y - GROUND) / 6000.0, -1.0, 1.0)),
                      gravity], np.float32)
    blocked = (code != 0) & (code != TUNNEL_CODE)
    end = np.array(road_end_inputs(raw, lane_x), np.float32)
    return np.concatenate([reduce_bands(code, axis=0).ravel(),
                           clear_path_features(blocked), state, end]).astype(np.float32).tolist()
