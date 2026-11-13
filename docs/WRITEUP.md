# Evolving a Skyroads pilot with NEAT

*A technical write-up of SkyroadsEvo: approach, methods, findings, struggles and results.*
*Work period: 10–13 November 2026. Status as of generation ~1,870 of the current run.*

## Summary

The goal was to evolve a pilot for *Skyroads* (Bluemoon, 1993) with evolutionary learning only:
no gradient descent, no imitation of human play, in the spirit of SethBling's MarIO. The pilot is a
NEAT neural network that sees the road ahead as a grid and presses steer, throttle and jump.

The project went through five training environments. Each was built to fix the main weakness of the
one before:

| # | Environment | Speed | What it taught us |
|---|---|---|---|
| 1 | **Starlane**, an original procedural simulator | ~3.5 s per generation | Fixed-topology GA, ES and NEAT all stall in the same local optimum; better inputs, not a better optimiser, broke it |
| 2 | **The real game in DOSBox**, screen then live memory | ~100 s per generation (40 pilots) | Exact game state and a correct finish signal matter more than anything else; real-time noise and slow generations cripple selection |
| 3 | **A sim matched to measured physics**, linked to DOSBox | ~5–9 s per generation | A hand-tuned sim never closed the sim-to-real gap |
| 4 | **The skyroads-sdl port, run headless** | ~20 ms per flight, ~500–1000× DOSBox | The game's own logic at simulator speed: the right environment |
| 5 | **Curriculum training in the port** | ~15–30 s per generation (500 pilots × 13–21 flights) | Most remaining problems were in the evolutionary dynamics, not the pilot |

**Results so far:**

- Roads 1 and 2 are mastered. About 470 of 500 pilots finish Road 2, and the best pilot finishes both every time.
- Road 3 is partly solved. The furthest pilot reaches row 52 of 60 (87%), but the best all-round pilot still stops at row 10.
- Pilots trained in the port also finish Road 1 in DOSBox, and have finished Road 2 there, though transfer is not reliable now that training no longer matches DOSBox timing.

The most useful findings were about evolution itself:

1. **Rank-based fitness hides the value of finishing.** A road finish ranked one place above falling off at 99%, so finishers never took over.
2. **Default NEAT mutation is far too destructive for fine control.** Only 8% of a finisher's mutated offspring kept the skill, and 2% after crossover; with gentle mutation, 96% did.
3. **Gentle mutation collapses diversity**, so it needs a share of strongly mutated "explorers" and active species control.
4. **Distance-only fitness is deceptive at multi-step obstacles.** Protected seeds and then MAP-Elites-style stepping-stone niches were needed to keep half-solutions alive.

---

## 1. Goal and ground rules

- **Learn by evolution only.** Selection, mutation and crossover over neural networks (NEAT, via `neat-python`).
  Shaping the task (curricula, practice starts, input features, reward terms) is allowed; hand-coding the
  behaviour, imitation learning and backpropagation are not.
- **Use the user's own copy of the game, unmodified.** Skyroads is freeware. Nothing from the game is
  committed to this repository, and the game's files are never modified.
- **Do not rebuild the real levels in a simulator.** We decided early not to recreate the game's levels in
  a custom sim, from its data files or from the screen. Simulators here use their own generated roads.
  The eventual answer was the existing third-party open-source port, which runs the game's own freeware
  data with its own logic.
- **Target:** one pilot that finishes Roads 1–3, ideally all 30.

## 2. The pilot

**Network.** NEAT, starting from a sparse direct input-to-output network (`initial_connection =
partial_direct 0.03`, no hidden nodes, tanh). NEAT adds nodes and connections as it goes. Population sizes
ranged from 40 (real-time DOSBox) to 500 (port).

**Inputs (165)**, built by one shared function (`starlane/realobs.py: build_obs`) used by every
environment after phase 2, so a pilot can move between them unchanged:

| Group | Count | Content |
|---|---|---|
| Road grid | 143 | 13 lanes centred on the ship (so the whole 7-lane road is visible from any lane) × 11 distance bands covering every row out to 20 ahead. Each cell is hole −1, road 0, tunnel 0.5 or block/deadly 1. Low blocks and deadly tiles read as clear when the ship is high enough to pass over them. |
| Clear-path features | 15 | For each of the 13 lanes, how far ahead it stays clear; the offset to the best lane; and that offset's direction. |
| Ship state | 5 | Position within the lane, position across the road, speed, height above road, gravity. |
| Road end | 2 | How close the road's end is, once in view, and where the final tunnel is relative to the ship. |

**Outputs (3).** Steer, throttle and jump.

- **Steering is proportional.** The output's size sets how often the key is held, as a duty cycle, which allows small corrections on narrow strips.
- **Throttle** has thresholds at ±0.33.
- **Jump** has two modes. In "hold", the key follows the output, as the game itself does. In "tap", one press per rising edge.

**Per-flight score.** `1000 × progress + 500 × end-approach + 1000 × finished`.

- **Progress** counts only while the ship is at road level, and is capped at 0.99 unless the road is finished.
- **End-approach** is the share of the last 5 rows before the final tunnel spent on the ground in the tunnel's lane.

## 3. Phase 1 — Starlane, an original simulator

We first built **Starlane**, an original, vectorised NumPy game with similar mechanics: 7 lanes, holes, low and
tall blocks, burning, boost, sticky and refuel tiles, fuel and oxygen, and procedurally generated levels at
adjustable difficulty. 200 pilots flew 16 fresh levels per generation in about 3.5 s.

**Fixed-topology networks (MLP, inputs → 32 → 16 → 3).**

- **The GA's self-adaptive mutation step collapsed** from 0.08 to about 0.01, and learning stalled. A floor of 0.02 fixed that.
- **In a 150-generation comparison, the GA beat OpenAI-style ES:**

  | Method | Easy held-out levels won |
  |---|---|
  | GA | 12/40 |
  | ES, step 0.05 | 7/40 |
  | ES, step 0.02 | 8/40 |

- **Widening the view didn't help:** 10–13/40. A hand-written rule pilot won 19/20, so the levels were beatable; the search wasn't finding the behaviour.

**NEAT.**

- **NEAT did better, with tiny networks:** 18–19/40 easy levels with about 16 hidden nodes and 19 links, against about 10,000 weights in the MLP.
- **Speciation needed help.**
  - At the default threshold only 2 species formed.
  - Lowering the threshold made every random starting genome its own species, which crashed reproduction.
  - An adaptive threshold aiming for a target species count, plus a minimum species size of 1, fixed it. More species didn't make learning faster.

**The first local optimum: bunny-hopping.**

- **Replays explained the plateau.** Every method stalled around 40–45%. The best pilots held full throttle and jumped almost constantly, which clears holes and low blocks for free but cannot get past walls.
- **Steering into a wall's gap was what they lacked.** It needs a left-versus-right comparison that small mutations rarely find from a pilot that already scores well.

**Fix: clear-path inputs.**

- **What they are:** for each lane, how far ahead the lane stays clear, plus the direction of the best lane. This is the same kind of preprocessing MarIO's tile grid did.
- **A trivial rule using only these inputs won 35–38/40.**
- **The NEAT vision pilot jumped from 19/40 to 30/40,** and was the first to climb the difficulty steps.

**Lesson:** with every method stuck at the same score, the bottleneck was the representation, not the optimiser.

## 4. Phase 2 — the real game in DOSBox

### 4.1 Screen reading failed

The Starlane vision pilot, driven through a screen-capture bridge, survived about 6 seconds on the real
Road 1.

- **Physics mismatch:** the sim's physics didn't match the real game.
- **Colour reading:** reading the road from colours only worked on the one road calibrated.
- **Tunnels:** a ship hidden in a tunnel looked like a fall.
- **Speed:** speed had to be guessed.

We switched to **evolving directly on the real game** in several DOSBox windows (6, then 12, then 18), at
about 4 minutes per generation of 40 pilots.

### 4.2 Reading live state (the MarIO way)

MarIO read Super Mario World's RAM. We did the same, read-only, from the running DOSBox process.

| What | Where in the emulated RAM |
|---|---|
| Ship forward position, lane position, height | ship struct at `0x8F38` |
| Speed | `0xC2D9` |
| Road cells ahead (7 per row) | from `0x9748` |

- **Cell encoding:** the low byte is the tile, with 0 a hole and `0x0C` deadly. The high byte is the block type: 1 a tunnel, 2–3 a low block, 4 a tall block.
- **Mislabelled tiles:** we initially had tunnel and low block swapped, so pilots had been shown tunnels as jumpable blocks.

**Finish signal.** A user observation was key: every road ends in a tunnel, and a road only finishes when
the ship drives into it at road level.

- **How a finish shows in memory:** entering the final tunnel snaps the height straight to 0. A fall sinks gradually.
- **Earlier wins were false:** early "wins" had been ships flying past the end.
- **Confirmed by test:** a scripted final approach produced the game's "Road Completed" screen.

**Gravity** varies by road, so each road gets one standing jump to measure it, and the result became an input.

### 4.3 Struggles on the real game

- **The hop habit came along from Starlane.** Replays showed jump held at 1.0 straight over the final tunnel.
  - **Fix 1:** a jump penalty plus the end-approach reward. Jump held fell from 88% to about 13%.
  - **Fix 2, later:** the penalty was dropped, since jumping is fine everywhere except the end, and the approach reward alone was kept.
- **The ending was too rare to learn.** No pilot reached it often enough.
  - **Fix: carrier practice.** The current best pilot flies the road, and each candidate takes over 12 rows before the end. Within 5 generations, 13 of 40 candidates drove the run-in and 4 finished for real.
  - **First unaided full-road win:** generation 107. That was followed by 64 full-road finishes in 52 generations.
- **Practice only at the end** left Roads 2 and 3 stuck at the same early spots: 18% and 25%.
  - **Fix: sticking-point practice.** Takeovers start 6 rows before where the furthest pilot stopped, so practice moves with the frontier.
- **Fitness balance.** Practice wins on Road 1 dominated, and one strong road was enough to survive.
  - **Fix:** fitness = 0.6 × mean + 0.4 × min of the per-road scores, plus a smaller practice term. Later, each road and practice point is ranked separately, so every road counts equally.
- **Lane geometry was off by about 0.44 of a lane.** Pilots misjudged their lane about half the time. Fixing it raised tunnel entries from about 45 to about 250 per generation.
- **Timing noise.** Decisions on an independent clock made the same pilot finish one flight and crash the next.
  - **Fix:** pilots decide once per game update (about 26 per second), synchronised by watching memory change.
  - **What remained:** real-time play remains noisy, which turns selection into a partial lottery with a population of only 40.
- **Throughput.** Fixing these cut a generation from about 240 s to about 85 s; it was still slow.
  - A faster capture mode took captures from 157 ms to 8 ms with 12 windows.
  - One process per window.
  - DOSBox fast-forward, with the speed-lock key rebound so it could be held.
  - Trimmed menu waits.
  - **Still slow:** about 42 generations an hour.

### 4.4 Result of phase 2

Road 1 was finished unaided many times. Road 2 was finished at least once, via a practice takeover; the
game's road menu showed its completion mark. Roads 2 and 3 never became reliable at real-time speed.

## 5. Phase 3 — a sim matched to measured physics

Memory access let us measure the real physics.

| Measurement | Real game | Matched sim |
|---|---|---|
| Time to top speed | 4.10 s | 4.07 s |
| Top speed | 6.03 rows/s | 6.03 rows/s |
| Jump peak (standard gravity) | 5195 | 5195 |
| Jump airtime | ~0.48–0.53 s | 0.50 s |

We also measured ship width, game update rate (26 per second), key latency (one update), and how a
ship falls: it sinks about 850 height units per row, so a fall covers about 12 more rows. That last
measurement changed scoring: progress must only count rows covered at road level, or falling far is
rewarded.

**The matched sim, with its own generated roads, trained about 60× faster but transferred poorly.**

- **First attempt:** the best sim pilot reached 54% of real Road 1 and never finished it, while a real-trained pilot finished.
- **Physics variation** and dense "gauntlet" courses didn't close the gap.
- **Two-way migration** between the sim and the real run, with gene renumbering, didn't close it either. Sim migrants mostly placed mid-table on the real game, and fragmented NEAT's species.
- **The hourly transfer check** showed no steady gain.

**Lesson:** collisions, tile edges and timing details each matter. Matching them by hand never ended.

## 6. Phase 4 — the skyroads-sdl port, headless

[skyroads-sdl](https://github.com/pedrocatalao/skyroads-sdl) is an existing open-source port that runs the
game's freeware data files with a reimplementation of its fixed-point game logic. We cloned it unchanged
(commit `04a65f9`) and wrote a small C driver, `ext/skyenv/sky_env.c`. The driver:

- **Runs the game without a window:** it runs the port's own `game()` loop, with video stubbed out, and calls a Python
  `ctypes` callback once per game frame. The callback receives the ship state and returns the keys.
- **Ends a flight** on crash, finish or a frame cap.
- **Can render frames on request:** a second build links the port's renderer, so a viewer can draw the flight.

`starlane/portenv.py` wraps it with exactly the DOSBox bridge's inputs and scoring.

- **Speed:** a full flight takes about 20 ms, against about 10 s in DOSBox. That's roughly 500–1000× faster, with the
  game's own physics, tile behaviour and finish rule.
- **Validation:** the port's state matched DOSBox on the measured quantities.
  - Top speed: internal 10922, against 8704 on DOSBox's speed gauge.
  - Jump peak: 5195 at gravity 8.
  - The port's lateral position equals the DOSBox lane position.
- **A live side-by-side viewer** (`watch_side_by_side.py`) flies the same pilot in DOSBox and in the port at the same moment.
- **Lane geometry fix:** with the port's exact lane geometry (left border 95·128, lane width 46·128, ship 29 px), we found the lane
  mapping was still slightly off and fixed it again. That forced a fresh start.

## 7. Phase 5 — training in the port

### 7.1 Setup

- **Population:** 500. Each pilot flies every current road once, plus up to 4 evenly spaced practice
  takeovers per road and one at the population's sticking point. Once some pilot can reach a road's end,
  two end-of-road takeovers are added, 12 and 6 rows before the end.
- **Fitness:** rank-based. Each road and practice point is ranked separately, from 0 to 1000. Fitness is
  0.6 × mean + 0.4 × min of the road ranks, plus 0.25 × the mean practice rank.
- **Curriculum:** start on Road 1 alone. The next road unlocks when at least 10 of the top 20 pilots finish
  every current road, for 5 generations running. Roads already mastered earn a **keep bonus**
  (600 × share finished), so they are not forgotten.
- **Species control:** a proportional step toward a target of 12 species.
- **Runtime:** workers run at below-normal priority, and `train_forever.ps1` resumes automatically after any crash.

### 7.2 What went wrong, and what fixed it

The port runs were restarted several times (`port_v1`–`v4`, then the current `port`) as problems surfaced.

**1. Starting from the DOSBox-trained pilots carried their habits over.** After 170 generations, Road 1
was finished on only 1–4 of 400 flights per generation. A fresh random population learned faster: 83% of
Road 1 within 40 seconds.

**2. Hold-to-jump habit.**

- **The habit:** about 88% of pilots held jump, which clears most of Road 1 but bounces over the final tunnel.
- **First fixes:** road-end inputs, end-of-road practice starts, then a "tap" jump mode, where one press means one jump.
  - The tap mode made Road 1 learnable: run `port_v4` mastered it, including real finishes in DOSBox.
  - But tap also made jumping itself hard to discover.
- **Final fix:** once matching DOSBox timing stopped being a goal, we went back to the game's own held jump, exact timing and a decision every update. That learned much faster.

**3. Forgetting.**

- **The problem:** adding Road 2 wiped out Road 1. Finishes fell from about 400 to about 10 per 38 generations.
- **Fix:** the keep bonus.
  - **Too strong:** a bonus that outranked everything (5000) blocked the pilots that cleared Road 2's first gap.
  - **What worked:** 600, which makes losing Road 1 entirely cost more than anything a new road can give, while one missed finish (150) can be outweighed by real progress.

**4. Road 2 finishers that never spread.**

- **The run:** Road 1 was mastered by about generation 88, and Road 2 was added. From generation 774 to 1409, a few pilots finished Road 2 every generation, about 4 of 500 on average. But the top-20 bar was never met for 5 generations in a row, and at generation 1410 the last finisher's line died out.
- **Cause 1, rank fitness flattens a finish:** a pilot that finished Road 2 ranked only one place above one that fell at 99%.
- **Cause 2, the carrier was the furthest pilot of the current generation:** when finishers died out, the end-of-road practice for Road 2 disappeared with them, which made the end harder to relearn.
- **Fixes:**
  - A **finish bonus** for the newest road: +400.
  - A **best-ever carrier**: the carrier is only replaced when another pilot gets further.
  - A **rollback** to generation 1100's checkpoint, where finishers still existed.
- **Result:** finishers survived and all of them made the top 20, but their number stayed flat at about 4–5.

**5. Mutation fragility, measured.**

- **The test:** we took the 4 pilots that finished Roads 1 and 2, bred 60 offspring from each, and flew them.

| How offspring were made | Still finish both roads |
|---|---|
| neat.cfg mutation (80% of weights perturbed by σ 0.4, 5% replaced, frequent structural changes) | **8%** |
| Crossover with a species-mate + the same mutation (what NEAT normally does) | **2%** |
| Gentle mutation (10% of weights, σ 0.1, no replacement, rare structure changes) | **96%** |
| Crossover with a species-mate + gentle mutation | 29% |
| Crossover between two finishers + gentle mutation | 45% |

- **Why it stalled:** at the default settings, finishers survived as unchanged elites but almost never passed the skill on, so their number settled at about 4.
- **Fix:** gentle mutation, with 75% of offspring bred from a single parent; neat-python always performs crossover, so its genome crossover is wrapped.
- **Result:** within 5 generations, Road 2 finishers went from 6 to 79, the top 20 all finished both roads, and Road 3 unlocked at generation 1265. Road 2 finishers later reached about 470 of 500.

**6. Diversity collapse.**

- **What happened:** gentle mutation made the population near-clones, 2 species at the threshold floor. All 500 pilots hit the same block just after Road 3's first gap, at row 10 (12%), for 155 generations: 347 drifted right into it and 114 drifted left.
- **Fix: explorers.** 20% of offspring are mutated with the original strong settings.
- **A second problem, species churn:** the explorers then caused species to explode from 2 to 105, one per explorer.
  - neat-python never merges species; they only die out after 25 stagnant generations. So the count stayed at 105 while the threshold climbed toward its cap.
  - **Fix:** whenever the count exceeds twice the target, everyone is re-sorted into species from scratch.
- **Result:** one explorer immediately got past the block, to 27%. But its Roads 1–2 skills were broken, so it ranked near the bottom and its line died out within generations.

**7. Deception.**

- **Protected seeds:** the 5 pilots that got furthest on the newest road are kept even if they fail older roads. They are planted back each generation, with 25 gently mutated children. Ranking is by newest-road progress, then old-road score, so the archive hill-climbs toward pilots that do both.
  - **Result:** Road 3 went from 27% to 86% in 4 generations. The seeds' Roads 1–2 score climbed from 1289 to 1700 out of 2500: finishing Road 1 and reaching 90% of Road 2.
- **A ranking bug:** comparing progress to 4 decimals let a seed with 86.49% and 407 old-road points outrank one with 86.38% and 1700. Fixed by ranking on whole rows.
- **The second wall (row 52 of 60):**
  - The left lane ends at row 49. The only way on is a jump right onto short middle lanes, then a second jump onto a far-right strip.
  - A pilot that makes the first jump and then falls scores the same as one that rides off the end, so the half-solution is always discarded.
  - Nothing changed for 240 generations.
- **Fix: stepping-stone niches (MAP-Elites).**
  - Each flight records the last cell the ship stood on (row, lane).
  - Each cell keeps the pilot that got there with the best old-road score.
  - The furthest niche in each lane is planted each generation.
  - An extra practice start 9 rows before the wall gives room to set up the crossing.
  - **Result:** within 70 generations the lane-2 stepping stone advanced from row 44 to row 49, and the right-hand lanes were explored to row 35 for the first time.

### 7.3 Current state (generation ~1,870)

| | Status |
|---|---|
| Road 1 | Mastered: about 270–490 of 500 pilots finish it each generation |
| Road 2 | Mastered: up to about 470 of 500 finish it; the best pilot finishes it every time |
| Road 3 | Best all-round pilot stops at row 10 (12%). Furthest pilot: row 52 of 60 (87%). The niche archive holds 24 niches, and the lane-2 line is advancing toward the crossing. |
| Species | Oscillates between about 5 and 20 under the re-sort rule |
| Speed | About 30 s per generation (500 pilots × 21 flights) |

## 8. Findings

1. **Exact state and a correct success signal first.** The biggest single gains came from reading the game's real
   state and from knowing exactly what a finish is. Screen reading and guessed signals produced false wins and
   hid the real problems.
2. **Environment speed and determinism decide what's feasible.** Real-time play gave about 40 generations an hour
   with noisy fitness. The headless port gives deterministic flights about 1000× faster. Many fixes that took hours to
   evaluate in DOSBox took minutes in the port.
3. **Don't hand-build a simulator of a game you can run.** The matched sim reached the measured physics but
   never transferred. Running the game's own logic (the open-source port) removed the sim-to-real gap.
4. **Plateaus at the same score across algorithms point at inputs or fitness, not the optimiser.** GA, ES and
   NEAT all stalled in the bunny-hop optimum; clear-path inputs broke it.
5. **Rank-based fitness is robust but flattens discontinuities.** A road finish needs an explicit bonus, or it
   is worth one rank.
6. **Measure mutation fragility directly.** One quick test (mutate a skilled genome, re-test the offspring)
   explained hundreds of stagnant generations. NEAT's typical settings suit exploration; holding a fine-control skill
   needs much gentler mutation and less crossover.
7. **Exploitation and exploration have to be separated explicitly.** Gentle mutation keeps skills but
   collapses diversity; strong mutation explores but destroys skills. A mixed offspring policy, plus species
   control that can actually merge species, keeps both.
8. **Distance-only fitness is deceptive at multi-step obstacles.** Protected seeds and then quality-diversity
   niches kept half-solutions alive long enough to be completed. This is the classic case for novelty or
   quality-diversity search.
9. **Practice starts placed at the frontier are a powerful curriculum.** Carrier takeovers placed where the population
   gets stuck gave every pilot repeated attempts at the hardest spot. They need a stable carrier, the best ever rather than the best this generation,
   and enough run-up to attempt the manoeuvre.
10. **Mid-run changes have a cost.** Each fix reset some of what evolution had settled into. Checkpoints
    every 10 generations and the ability to roll back were essential.

## 9. Limitations

- **One network for all roads.** The curriculum asks a single pilot to keep every earlier road while learning the next, and
  the keep bonus and seeds are hand-tuned trade-offs.
- **Port training no longer matches DOSBox timing.** It uses exact timing, held jump and a decision every update, so DOSBox
  transfer is mixed. In the side-by-side viewer, Road 1 finishes in both; Road 2 sometimes fails in DOSBox.
- **Shaping is ours, not evolution's.** The clear-path inputs, road-end inputs, end-approach reward, practice starts and niches
  are design choices. The pilots still learn only through selection and variation.
- **Roads 4–30 are untested.** Later roads include gravity up to 20 on Road 20, where jumping is impossible, as well as
  fuel and oxygen limits and tile types not yet mapped.
- **Not tuned systematically.** Hyperparameters were not tuned in any principled way; most changes were made in response to observed failures.

## 10. Future work

- **Let the niches finish Road 3,** then watch whether the stepping-stone lineage wins Roads 1–2 back and unlocks Road 4.
- **Breed seeds with finishers:** crossover between niche seeds and the main population's old-road finishers. The test showed finisher × finisher
  crossover keeps the skill 45% of the time.
- **Specialists:** per-road specialist populations as "rescue teams", fed into the main run when it stalls on a road.
- **Recover DOSBox transfer:** re-enable timing jitter and per-frame decisions once roads are mastered, and fine-tune for DOSBox.
- **Deceptive spots:** novelty search, rewarding new final positions, as an alternative to hand-chosen niche dimensions.

## Appendix A — repository map

| Path | What it is |
|---|---|
| `starlane/` | Starlane sim (`sim.py`, `level.py`), shared real-game inputs (`realobs.py`), matched sim (`realsim.py`, `realroads.py`), port wrapper (`portenv.py`), NEAT helpers |
| `bridge/` | DOSBox window control, screen capture, read-only memory reading (`memory.py`), live flying (`live.py`) |
| `ext/skyenv/` | Headless driver for the skyroads-sdl port (`sky_env.c`, `build.sh`) |
| `train.py`, `train_neat.py` | Phase 1 trainers (Starlane) |
| `train_real.py`, `train_realsim.py` | Phase 2–3 trainers (DOSBox, matched sim) |
| `train_port.py`, `train_forever.ps1` | Phase 5 trainer, and its auto-resume wrapper |
| `watch_side_by_side.py`, `check_transfer.py` | DOSBox vs port viewer, hourly transfer check |

## Appendix B — current training settings (`train_port.py` defaults)

| Setting | Value |
|---|---|
| Population | 500 |
| Flights per road | 1, exact timing, held jump, decision every game update |
| Practice | 4 evenly spaced takeovers + reach − 3 + reach − 9; end-of-road takeovers at 12 and 6 rows |
| Fitness | rank per road and practice point: 0.6 × mean + 0.4 × min + 0.25 × practice mean |
| Keep bonus (mastered roads) | 600 × share finished |
| Finish bonus (newest road) | 400 |
| Unlock | ≥ 50% of the top 20 finish every current road, 5 generations running |
| Mutation | gentle (10% of weights, σ 0.1); 20% explorers with neat.cfg settings |
| Breeding | 75% single parent |
| Species | target 12, threshold floor 0.05, re-sort when above 24 |
| Seeds | furthest stepping-stone niche in each of up to 7 lanes, plus 25 gentle children |
