# SkyroadsEvo

For the story of the project (approach, methods, findings, struggles and results), see
[docs/WRITEUP.md](docs/WRITEUP.md).

Neuroevolution pilots for a space-road racing game. There are two parts:

1. **Starlane**: an original, fast, headless simulator with road-racer mechanics. It has a
   7-lane road floating in space, holes to jump, low slabs to jump onto, tall walls and pillars to
   steer around, burning, boost, sticky and refuel tiles, plus fuel and oxygen limits. Levels are
   generated procedurally, so the pilot can't memorise them; it has to generalise. Evolution runs
   hundreds of pilots in parallel on the same level using vectorised numpy.
2. **Bridge**: drives *your own copy* of a real DOS game running in DOSBox. It captures the screen,
   turns it into the same observation the simulator produces, and sends key presses.

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Train

```bash
.venv\Scripts\python train.py --out runs/full
.venv\Scripts\python train.py --obs vision --noise 0.05 --out runs/vision
```

* `--obs full` lets the pilot see tile heights and kinds, plus its own height, vertical speed,
  fuel and oxygen. It's the best pilot for the simulator.
* `--obs vision` gives only what a screen reader can plausibly extract: hole / floor / raised for
  each sampled cell, lateral position, estimated speed, and time since the last jump. **Use it for the
  real-game bridge.** `--noise` adds observation noise so the pilot tolerates imperfect vision.
* Training uses a **curriculum** that starts at difficulty 0 (see below).
* `--resume` continues from `runs/<name>/population.npz`. Ctrl+C stops cleanly.

Each run directory gets `log.csv`, `best.npz` (best pilot at the hardest difficulty reached),
`latest_best.npz` and `population.npz`.

### The algorithm

`starlane/evolve.py` evolves the weights of a fixed MLP (`obs → 32 → 16 → 3`, tanh). Each
generation, every candidate flies the same 16 fresh random levels. Fitness is the mean of
`1000 × progress + 1000 × win`. There's no reward for speed: the goal is just to reach the end, and oxygen, which runs out with time, punishes going too slowly. Pick the algorithm with `--algo`:

* **`es`: evolution strategy**, in the style of OpenAI's ES. There's one mean genome. Each
  generation it samples 99 Gaussian perturbations, each tried both added and subtracted (antithetic
  pairs). It ranks all 199 candidates and moves the mean toward the better ones using Adam (`--sigma`,
  `--lr`). The unperturbed mean flies the same levels too: it's the saved pilot, and its win rate
  drives the curriculum.
* **`ga` (default): genetic algorithm.** In a 150-generation comparison it beat `es` (12/40 vs 7–8/40 easy levels won).
  * The top 8 carry over unchanged, and parents are chosen by tournament selection (size 5).
  * 30% of children come from **neuron-level crossover**: whole neurons, meaning their incoming
    weights plus bias, are taken from one parent or the other.
  * Mutation is Gaussian with a **self-adaptive step size**, floored at 0.02 so it can't collapse.

**NEAT** (`train_neat.py`, the algorithm MarIO uses, via `neat-python`):

* Networks start tiny: each genome gets a few random input-to-output links and no hidden neurons.
* Mutation adds and removes links and neurons, as well as changing weights.
* Genomes are grouped into species by structural similarity, so a new structure competes mainly
  within its own species while its weights get tuned.
* Uses the same levels, fitness and curriculum as above. It saves `best.pkl` and `latest_best.pkl`,
  plus `neat-checkpoint-*` every 10 generations (`--resume`).

```bash
.venv\Scripts\python train_neat.py --out runs/neat_full
.venv\Scripts\python train_neat.py --obs vision --noise 0.05 --out runs/neat_vision
```

`watch.py`, `evaluate.py` and `bridge/play_real.py` accept either kind of pilot file.

The curriculum raises difficulty by 0.05 whenever a smoothed (moving-average) win rate reaches
`--promote` (default 60%).

Outputs are thresholded into steer {-1, 0, 1}, throttle {-1, 0, 1} and jump.

## Watch / play

```bash
.venv\Scripts\python watch.py runs/full/best.npz --difficulty 0.5
.venv\Scripts\python watch.py --human
.venv\Scripts\python evaluate.py runs/full/best.npz
```

Viewer keys: **R** restart, **N** next level, **+/-** difficulty, **F** fast-forward, **Esc** quit.
Human controls: arrows to steer and throttle, Space to jump.

## Training in the skyroads-sdl port (current)

[skyroads-sdl](https://github.com/pedrocatalao/skyroads-sdl) is an existing open-source port of Skyroads
that runs the game's own freeware data with its original fixed-point physics. It is cloned unchanged into
`ext/skyroads-sdl` (data copied from `game/skyroads` into `ext/skyroads-sdl/data`). `ext/skyenv/sky_env.c`
drives its game logic with no window: once per game frame it gives Python the ship state and takes back the
keys. `starlane/portenv.py` wraps it with the same inputs and scoring as the DOSBox bridge, so pilots move
between the two unchanged. Note the port's LICENSE: the parts derived from Bluemoon's code carry no license.

```bash
# once: get the port (not part of this repo) and copy your game's data files into it
git clone https://github.com/pedrocatalao/skyroads-sdl ext/skyroads-sdl   # tested at 04a65f9
# copy the .DAT, .LZS, .REC and .SND files from game/skyroads into ext/skyroads-sdl/data
# once: MSYS2 (C:\msys64) with mingw-w64-ucrt-x86_64-gcc, then from a UCRT64 shell
bash ext/skyenv/build.sh
# train (about 3 s per generation of 200 pilots)
.venv\Scripts\python train_port.py --seed-from runs/real --out runs/port
.venv\Scripts\python train_port.py --resume --out runs/port
# hourly check in DOSBox at normal speed, with the port's prediction alongside
.venv\Scripts\python check_transfer.py
```

* The game reads controls once per drawn frame (about 26 a second in DOSBox) and runs its physics at 36
  ticks a second. Training flights jitter each frame between 1 and 2 ticks as DOSBox does.
* Each pilot flies each road twice and practises from several points along every road, plus takeovers 12
  and 6 rows before each reachable road's end (ranked like whole roads); fitness is its standing on each
  road and practice point. Training starts on Road 1 alone and adds the next road once the best pilot
  has finished every current road on every flight for 3 generations running (`--unlock-after`). Roads
  already mastered earn a bonus for the share of their flights finished (`--keep-weight`, default 600):
  forgetting them costs more than a new road can give, but one missed finish can be outweighed.
* Jump is a press, not a held key: the jump output switching on gives one jump, and it must switch off
  before the next. Two inputs describe the road's end once it is in view (how close; where the tunnel is).
* Training processes run at below-normal priority so a DOSBox check window still gets CPU time.
* Since the goal is now finishing roads rather than matching DOSBox timing, training defaults to exact
  timing (no jitter, one flight per road), a decision every game update (`--frame-ticks 1`) and the game's
  own held jump (`--jump hold`). `--jitter --flights 4 --frame-ticks 1.385 --jump tap` restores the
  DOSBox-matched setup. `train_forever.ps1` runs training and resumes it automatically if it ever stops.
* `--fresh` starts from new random pilots. The current `runs/port` run was started fresh after the lane
  mapping in `starlane/realobs.py` was corrected to the game's own geometry (the earlier run is `runs/port_v1`).
* `watch_side_by_side.py` plays the latest best pilot live in DOSBox and in the port, next to each other
  on the second monitor.

## Matched simulator, linked to the real game (earlier approach)

`train_realsim.py` trains in a simulator tuned to the real game's measured physics (`starlane/realsim.py`),
with the same inputs pilots get on the real game (`starlane/realobs.py`) and its own generated roads
(`starlane/realroads.py`). A generation takes seconds instead of minutes.

Each sim road gets slightly different physics (`--vary`, from the road's seed so every pilot on it is
treated alike), and each generation adds full-difficulty gauntlet courses packed with every kind of
challenge (`--gauntlets`). Progress only counts rows covered at road level, in the sim and the real game.

The two runs swap pilots through `migrants.pkl` files (`starlane/migrate.py`). The real run keeps a nursery:
each new sim pilot starts a line that holds a few reserved places for several generations while it adapts
(`--nursery-lines/-slots/-gens`). The sim brings in the real game's best every 10 generations. Incoming pilots
are renumbered so they can breed in the receiving run. The real run ranks pilots on each road separately and
averages those standings, so every road counts the same.

`check_transfer.py` measures whether sim training carries over: every hour it flies the sim's top pilots and
the real run's latest best on real roads 1-3 and appends the results to `runs/logs/transfer.txt`.

```bash
.venv\Scripts\python train_realsim.py --out runs/realsim
.venv\Scripts\python train_real.py --resume --out runs/real          # links to runs/realsim by default
.venv\Scripts\python check_transfer.py                               # hourly sim-to-real check
```

## Real game (DOSBox)

You need your own copy of the game. Put its files in `game/skyroads/` and run `game\run_skyroads.bat` to start it in the portable DOSBox Staging in `game/dosbox/`. Nothing from the game is committed to this repo.

1. Train a vision pilot (see above).
2. Run the game, get a straight full-width road on screen, then calibrate:
   ```bash
   .venv\Scripts\python -m bridge.calibrate --window SKYROADS.EXE
   ```
   Click the 4 road corners (near-left, near-right, far-right, far-left), then empty-space colours,
   then the ship, then (optionally) raised-block colours. After that, enter how many rows ahead the
   near and far edges are. This writes `bridge/calibration.json`, where you can also remap keys and
   tune `bg_tol`, `ship_tol` and `speed_scale`.
3. Dry run first. It shows what the pilot sees and would press, without sending keys:
   ```bash
   .venv\Scripts\python -m bridge.play_real runs/vision/best.npz --dry-run
   ```
4. Run live. Capture and key presses go straight to the game window, so it can sit behind other
   windows and your keyboard stays free. **Hold F12 to stop.** Add `--seconds 30 --no-debug
   --save-frames some/folder` to record annotated frames. The game restarts a road by itself after
   the ship is lost.

For the cleanest colours, `game/skyroads.conf` turns DOSBox's CRT shader off. `bridge/calibration.json`
can also list exact road colours under `floor` (anything else reads as a hole) and block colours
under `raised`. Those lists are specific to a road's palette.

## Train on the real game

`train_real.py` evolves pilots by playing the actual game in several DOSBox windows, on its real
roads. Nothing reads or changes the game's files; it only uses the screen and key presses.

```bash
.venv\Scripts\python train_real.py --windows 6 --out runs/real
.venv\Scripts\python train_real.py --resume --out runs/real
.venv\Scripts\python train_real.py --play runs/real/best.pkl --roads 0 1 2
```

* It starts from the latest NEAT vision pilots in `runs/clear_vision`, keeping the best on simulated levels.
* Each flight scores `1000 × progress + 500 × last-tunnel approach + 1000 × finished`. A pilot's fitness is `0.6 × mean over roads + 0.4 × its worst road + 0.25 × practice flights`, so
  one strong road isn't enough. The approach score is the share of the 5 rows before the last tunnel spent on
  the ground in its lane; a road is finished by driving into that tunnel at road level.
* During a flight the trainer reads the running game's memory (read-only, see `bridge/memory.py`) for the
  ship's position, height, speed and the road cells ahead, the way MarIO read its emulator's RAM. Falls,
  crashes and finishes come straight from those values. The screen is only used for menus and as a backstop.
* Each pilot also gets a practice flight on each road in `--practice-roads`: the pilot that got furthest on that
  road last generation flies it, and the candidate takes over 6 rows before where that pilot stopped (12 rows
  from the end once a road can be finished). So practice always sits at the population's sticking point.
  Gravity is measured with one standing jump per road and given to the pilot as an input.
* `--fast` runs the games fast-forwarded (about 10x) by holding DOSBox's speedlock key, rebound to F11 in
  `game/skyroads_fast.map`. Pilots still decide once per game update, so they fly the same way; each fast window
  keeps a CPU core busy, so use fewer windows (about 10 on a 24-thread CPU).
* Pilots decide once per game update (about 26 a second), by waiting for the ship's state to change in memory.
  Deciding on an independent clock made the same pilot finish Road 1 in one flight and crash in the next.
* Steering is proportional (`starlane/realobs.py`, `Steering`): the size of the steering output sets how often the
  key is held, so pilots can make small corrections on narrow strips. Same in the real game and the matched sim.
* Road numbers run 0–29 (three per world, in menu order). A pilot that finishes every current road
  unlocks the next one.
* Windows go on a second monitor when there is one. Hold F12 or press Ctrl+C to stop.
* It runs in real time, so a generation takes minutes.

Known limitations of the bridge (it's a starting point, not a finished transfer):

* It assumes a camera fixed relative to the road edges. If the game's camera slides sideways with
  the ship, calibration will drift.
* Speed isn't read from the screen. It's dead-reckoned from the pilot's own throttle commands and
  the simulator's acceleration, scaled by `speed_scale`.
* Simulator physics (jump arc, top speed, steering rate) won't match the real game exactly. Expect
  to tune `starlane/config.py` to match what you see, retrain, and possibly fine-tune with
  real-game runs.
* Telling floor from raised blocks needs the optional `raised` colours. Without them, blocks read as floor.

## Layout

```
starlane/config.py   physics constants
starlane/level.py    procedural level generator
starlane/sim.py      vectorised physics + observations + fitness
starlane/policy.py   batched MLP policies, genome save/load
starlane/evolve.py   GA + ES + parallel evaluator
starlane/neat_evo.py NEAT config + batched evaluation
starlane/pilot.py    loads any saved pilot (.npz or .pkl)
starlane/render.py   pygame chase-cam renderer
train.py / train_neat.py / watch.py / evaluate.py
bridge/              DOSBox screen capture, calibration, live driving
```
