"""Fly pilots in the skyroads-sdl port's own game logic, headless and much faster than DOSBox.

The port (ext/skyroads-sdl, an existing open-source port by Pedro Catalão) runs the game's own
freeware data with its original fixed-point physics. ext/skyenv/skyenv.dll drives it with no
window: once per game frame it hands us the ship state and takes back the keys to hold.

Pilots get exactly the inputs they get in DOSBox (starlane/realobs.build_obs): the ship's lateral
position and height in the same units, the same road cells, and speed rescaled to DOSBox's speed
gauge. Scoring matches bridge/live.fly, so results are comparable and pilots move freely between
the two.
"""
import ctypes
import os

import numpy as np

from . import config as C
from .realobs import GROUND, REFERENCE_JUMP, JumpTap, Steering, build_obs
from .sim import OBS_ROWS

HERE = os.path.dirname(os.path.abspath(__file__))
DLL = os.path.join(HERE, "..", "ext", "skyenv", "skyenv.dll")
DATA = os.path.join(HERE, "..", "ext", "skyroads-sdl", "data")

K_LEFT, K_RIGHT, K_UP, K_DOWN, K_SPACE, K_ESC = 1, 2, 4, 8, 16, 32
FINISHED, CRASHED, TIMEOUT = 1, 2, 3
CRASH_WHY = {1: "crashed", 2: "burned", 3: "fell", 4: "fuel", 5: "oxygen", 6: "crashed"}
MAX_SPEED = 10922                # the port's top speed (6 rows/s at 36 ticks/s)
DOSBOX_SPEED_FULL = 8704         # what DOSBox's speed variable reads at top speed
START_ROW = 3
BELOW_ROAD = 300                 # deeper than this under road level the ship can't recover
HANDOVER_ROWS = 12
APPROACH_ROWS = 5
LEFT_BORDER, SLAB_WIDTH = 95 * 128, 46 * 128     # the game's lane geometry (lateral units)
FRAME_TICKS = 36.0 / 26.0        # DOS game: controls read ~26 times a second, physics at 36


class State(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("z", ctypes.c_uint32),
                ("vx", ctypes.c_int32), ("vz", ctypes.c_int32), ("crash", ctypes.c_uint32),
                ("fuel", ctypes.c_uint32), ("oxy", ctypes.c_uint32), ("frame", ctypes.c_uint32),
                ("tick", ctypes.c_uint32)]


POLICY = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.POINTER(State))
_LOADED = {}                     # DLL path -> initialised library (one per process)


def jump_peak(gravity):
    """Peak height above road level of a standing jump, stepping the game's own jump arithmetic."""
    acc = -((45 * 128 * gravity) // 400)
    vz, peak = 4 * 45 * 128 // 20, 0
    while True:
        vz += acc
        if vz <= 0:
            return peak
        peak += vz


def game_lane(y):
    """The lane the game itself puts the ship's centre in (0-6), from its lateral position."""
    return int((y - LEFT_BORDER) // SLAB_WIDTH)


class PortEnv:
    """One per process (the game logic keeps global state)."""

    def __init__(self, dll=DLL, data=DATA):
        path = os.path.abspath(dll)
        # Windows loads a DLL once per process, and the game's memory arena is fixed: initialise the
        # game data only the first time, or a second PortEnv in the same process runs out of memory
        if path in _LOADED:
            self.lib = _LOADED[path]
            self.road = None
            return
        self.lib = ctypes.CDLL(path)
        self.lib.env_fly.argtypes = [POLICY, ctypes.c_double, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.POINTER(State)]
        self.lib.env_road_cells.restype = ctypes.POINTER(ctypes.c_uint16)
        if hasattr(self.lib, "env_frame_rgb"):          # builds from before it was added lack it
            self.lib.env_frame_rgb.argtypes = [ctypes.c_void_p]
        if self.lib.env_init(os.path.abspath(data).encode()) != 0:
            raise RuntimeError("could not load the game data")
        _LOADED[path] = self.lib
        self.road = None

    def load(self, road):
        """road: 0-based like the rest of the project (0 = Red Heat Road 1)."""
        if road != self.road:
            length = self.lib.env_load_road(road + 1)
            if length <= 0:
                raise RuntimeError(f"could not load road {road}")
            n = length + OBS_ROWS + 8
            raw = np.ctypeslib.as_array(self.lib.env_road_cells(), shape=(n * 7,))
            self.cells = raw.reshape(n, 7).copy()
            self.cells[length + 1:] = 0
            self.length = length
            self.gravity = self.lib.env_gravity()
            factor = REFERENCE_JUMP / max(jump_peak(self.gravity), 1)
            self.grav_in = float(np.clip((factor - 1.0) * 5.0, -1.0, 1.0))   # as bridge/live measures it
            self.road = road
            tail = self.cells[max(0, length - 4):length]
            rows, lanes = np.nonzero((tail >> 8) == 1)
            self.tunnel_end = ((max(0, length - 4) + int(rows.min()), set(int(v) for v in lanes))
                               if len(rows) else None)
        return self.length

    def fly(self, net, road, carrier=None, handover_row=None, frame_ticks=FRAME_TICKS, max_seconds=150.0,
            seed=0, on_frame=None, jump="tap"):
        """One attempt. Returns a dict shaped like bridge/live.GameWindow.fly's.
        seed: 0 = even frame timing (repeatable); otherwise each frame covers 1 or 2 game ticks at
        random, as in DOSBox, so a pilot can't rely on exact timing.
        on_frame(state): optional, called every frame before the pilot decides (e.g. to draw it).
        jump: "tap" (one jump per switch-on of the output) or "hold" (the key follows the output, so
        holding it re-jumps on landing, as the game itself does)."""
        length = self.load(road)
        handover = length - HANDOVER_ROWS if handover_row is None else min(handover_row, length - 3)
        handover = max(START_ROW, handover)
        cells, grav_in, tunnel_end = self.cells, self.grav_in, self.tunnel_end
        steering = Steering()
        tap = JumpTap()
        jump_mode = jump
        last_z = [GROUND]
        m = {"handed": carrier is None, "progress": 0.0, "approach": 0, "good": 0, "jumps": 0, "ticks": 0,
             "best": 0.0, "best_frame": 0, "tunnels": 0, "was_tunnel": False, "reached": False, "stalled": False}
        origin = START_ROW if carrier is None else handover

        def policy(sp):
            s = sp.contents
            if on_frame is not None:
                on_frame(s)
            pos = s.x / 65536.0
            row = int(pos)
            if not m["handed"] and pos >= handover:
                m["handed"] = True
            level = s.z >= GROUND - BELOW_ROAD
            if m["handed"] and level:      # 100% only for a real finish, not for flying past the end
                m["progress"] = max(m["progress"], min(0.99, (pos - origin) / max(1, length - origin)))
            if level and pos >= length - 1.5:
                m["reached"] = True
            lane = min(max(game_lane(s.y), 0), 6)
            if m["handed"] and s.z == last_z[0] and s.z >= GROUND:
                m["rest"] = (row, lane)        # last spot the ship stood on (road or block top)
            in_tunnel = (int(cells[row, lane]) >> 8) == 1
            m["tunnels"] += in_tunnel and not m["was_tunnel"] and s.z <= GROUND + 400
            m["was_tunnel"] = in_tunnel
            if tunnel_end and tunnel_end[0] - APPROACH_ROWS <= pos < tunnel_end[0]:
                m["approach"] += 1
                m["good"] += s.z <= GROUND + 400 and lane in tunnel_end[1]
            if pos > m["best"] + 1e-4:
                m["best"], m["best_frame"] = pos, s.frame
            elif s.frame - m["best_frame"] > (39 if m["best"] > START_ROW + 0.05 else 130):
                m["stalled"] = True                # 1.5 s without moving on (5 s at the start)
                return K_ESC
            speed = s.vx * DOSBOX_SPEED_FULL / MAX_SPEED
            obs = build_obs(cells[row:row + OBS_ROWS], s.y, s.z, speed, grav_in)
            out = (net if m["handed"] else carrier).activate(obs)
            steer = steering(out[0])
            thr = 1 if out[1] > 0.33 else (-1 if out[1] < -0.33 else 0)
            jump = out[2] > 0.5 if jump_mode == "hold" else tap(out[2], s.z > last_z[0] and s.z > GROUND)
            last_z[0] = s.z
            if m["handed"]:
                m["jumps"] += jump
                m["ticks"] += 1
            return ((K_LEFT if steer < 0 else 0) | (K_RIGHT if steer > 0 else 0) | (K_UP if thr > 0 else 0)
                    | (K_DOWN if thr < 0 else 0) | (K_SPACE if jump else 0))

        out = State()
        cb = POLICY(policy)
        result = self.lib.env_fly(cb, ctypes.c_double(frame_ticks), int(max_seconds * 26), seed & 0xFFFFFFFF,
                                  ctypes.byref(out))
        if result == FINISHED:
            why = "finished"
        elif result == CRASHED:
            why = CRASH_WHY.get(out.crash, "crashed")
        else:
            why = "stalled" if m["stalled"] else "timeout"
        won = why == "finished"
        end_score = m["good"] / m["approach"] if m["approach"] else 0.0
        if won:
            end_score = 1.0
        return {"progress": 1.0 if won else m["progress"], "won": won, "why": why, "end_score": end_score,
                "seconds": out.tick / 36.0, "hz": 26.0, "tunnels": int(m["tunnels"]), "cell": 0,
                "length": length, "reached_end": m["reached"] or won,
                "jump_frac": m["jumps"] / max(m["ticks"], 1), "skipped": not m["handed"],
                "row": out.x / 65536.0, "rest": m.get("rest", (int(origin), 3))}
