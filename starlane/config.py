"""Physics and game constants for the Starlane simulator.

Units: one lane is 1.0 wide, one road row is 1.0 long, time is in seconds.
"""

LANES = 7            # road width in lanes
DT = 1.0 / 30.0      # simulation tick

BLOCK_H = 0.6        # height of a low block (jumpable); walls are TALL_H
TALL_H = 3 * BLOCK_H  # walls / pillars: too tall to jump onto
GRAVITY = 22.0
JUMP_V = 7.5         # initial upward velocity of a jump (apex ~1.28, airtime ~0.68 s)

V_MAX = 14.0         # top forward speed (rows / s)
ACCEL = 9.0
BRAKE = 14.0
COAST_DRAG = 0.0     # no drag: the ship keeps its speed when coasting
STEER_SPEED = 5.0    # lateral speed (lanes / s)
CRASH_SPEED = 5.0    # hitting a wall faster than this destroys the ship

HALF_W = 0.30        # ship half width
HALF_L = 0.40        # ship half length
STEP = 0.12          # tolerance when landing / touching wall tops
FALL_DEATH_Y = -2.0
STALL_SECONDS = 3.0  # no forward progress for this long ends the run

GAP = -100.0         # "height" of a missing tile

# tile kinds (only meaningful where there is a surface)
K_NORMAL, K_BOOST, K_STICKY, K_BURN, K_FUEL = 0, 1, 2, 3, 4

# end-of-run reasons
RUNNING, WIN, FELL, CRASHED, BURNED, NO_FUEL, NO_OXYGEN, STALLED, TIMEOUT = range(9)
REASONS = ["running", "win", "fell", "crashed", "burned", "no fuel",
           "no oxygen", "stalled", "timeout"]
