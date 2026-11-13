# Teaching a Neural Network to Play Skyroads with Evolution

Skyroads is a 1993 DOS game where you fly a ship down a road floating in space. You steer, speed up, slow
down and jump over holes and blocks, and every road ends in a tunnel you have to drive into. I wanted to see
if an evolutionary algorithm could learn to play it, the same way SethBling's MarIO learned Super Mario World.
No gradient descent, no copying human play. Just breed the networks that fly furthest and let mutation do
the rest.

I built this over four days (10 to 13 November 2026), with Claude Code writing most of the code while I
directed it. This write-up covers what we tried, what broke, what fixed it, and where it stands now.

## The Goal

One pilot that can finish Roads 1 to 3, and ideally all 30 roads.

The rules I set:

- Learning happens through evolution only: selection, mutation and crossover. Shaping the task is fine
  (practice starts, extra inputs, reward terms). Hand-coding the behaviour or training on human play is not.
- The game runs from my own copy, unmodified. Skyroads is freeware, but none of its files are in this repo.
- No rebuilding the real levels inside a custom simulator.

## The Pilot

Each pilot is a NEAT network (neat-python). It starts tiny, with a few random links from inputs to outputs
and no hidden neurons, and NEAT adds links and neurons over time.

**Inputs (165):**

- **Road grid (143):** 13 lanes centred on the ship by 11 distance bands, covering every row up to 20 ahead.
  Each cell is a hole (-1), road (0), tunnel (0.5) or block/deadly tile (1).
- **Clear path (15):** for each lane, how far ahead it stays clear, plus which way the best lane is.
- **Ship state (5):** position in the lane, position across the road, speed, height and gravity.
- **Road end (2):** how close the end of the road is, and where the final tunnel is.

**Outputs (3):** steer, throttle and jump. Steering is proportional, so the size of the output sets how often
the key is held. That lets the pilot make small corrections on narrow strips.

**Score per flight:** `1000 × progress + 500 × end approach + 1000 × finished`. Progress only counts while the
ship is at road level, and is capped at 99% unless the road is actually finished. End approach is the share of
the last 5 rows spent on the ground in the tunnel's lane.

## Attempt 1: A Simulator of My Own

To start, we built Starlane, an original simulator with similar mechanics: 7 lanes, holes, low and tall blocks,
special tiles, fuel and oxygen, and randomly generated levels. A generation of 200 pilots flying 16 levels
took about 3.5 seconds.

For a base case, we evolved a fixed network (inputs → 32 → 16 → 3) with a genetic algorithm and with an
evolution strategy. Results on 40 easy levels the pilots had never seen:

| Method | Levels won |
|---|---|
| Genetic algorithm | 12/40 |
| Evolution strategy, step 0.05 | 7/40 |
| Evolution strategy, step 0.02 | 8/40 |

Switching to NEAT got 18 to 19 out of 40, using about 19 links instead of 10,000 weights. Better, but every
method stalled at the same place.

Replaying the crashes showed why. The pilots had learned to hold full throttle and jump constantly. Bunny
hopping clears holes and low blocks for free, so it scores well, but it can't get through walls. Steering
into a gap needs the network to compare its left and right sides, and small mutations rarely find that.

The fix was better inputs. I added clear-path features: for each lane, how far ahead it stays clear. With
those, the best vision pilot went from 19/40 to 30/40 and was the first to climb the difficulty levels.

**Lesson:** when every algorithm gets stuck at the same score, the problem is the inputs, not the algorithm.

## Attempt 2: The Real Game in DOSBox

The Starlane pilot lasted about 6 seconds on the real Road 1. The physics were different, reading the road
from screen colours only worked on one road, and a ship inside a tunnel looked like a crash.

So we moved training onto the real game, running in 6, then 12, then 18 DOSBox windows at once.

### Reading the game's memory

MarIO read the game's RAM, so we did the same. The trainer reads the running game's memory (read only) for the
ship's position, height, speed and the road cells ahead. That gave exact falls, crashes and finishes.

Two things I pointed out made a big difference:

- **Every road ends in a tunnel, and you have to drive into it at road level to finish.** Until then, pilots
  flying off the end of the road were being counted as wins. Entering the tunnel snaps the ship's height straight
  to 0, while a fall sinks slowly, so that became the finish signal.
- **Some roads have different gravity.** Each road now gets one standing jump to measure it, and gravity became
  an input.

### What went wrong, and what fixed it

- **The hopping came along from the simulator.** Replays showed the jump key held the whole way over the final
  tunnel. A small jump penalty plus the end approach reward dropped jump time from 88% to about 13%. Later I
  removed the penalty, since jumping is fine everywhere except the end.
- **Nobody reached the end often enough to learn it.** So we added practice: the best pilot flies the road, and
  each new pilot takes over 12 rows before the end. Within 5 generations, 4 of 40 were finishing for real.
  The first pilot to finish a whole road on its own came at generation 107.
- **Roads 2 and 3 were stuck at 18% and 25%.** Practice only covered the ending. Moving the takeover point to
  just before wherever the furthest pilot got stuck fixed that, because practice moved with the population.
- **The lane mapping was off by almost half a lane.** Pilots were wrong about which lane they were in about half
  the time. After the fix, tunnel entries went from about 45 to about 250 per generation.
- **Results were noisy.** The same pilot could finish one flight and crash the next. Syncing decisions to the
  game's own updates helped, but real time play stayed noisy, and with only 40 pilots, selection was partly luck.
- **It was slow.** Faster screen capture, one process per window, DOSBox fast forward and shorter menu waits took
  a generation from about 240 seconds down to about 85. That's still only about 42 generations an hour.

Road 1 got finished many times. Road 2 was finished once, in a practice flight, and the game's road menu showed
it. Neither was reliable.

## Attempt 3: A Simulator Matched to the Real Game

With memory access, we could measure the real physics and build a simulator that matched them:

| Measurement | Real game | Matched sim |
|---|---|---|
| Time to top speed | 4.10 s | 4.07 s |
| Top speed | 6.03 rows/s | 6.03 rows/s |
| Jump height | 5195 | 5195 |
| Time in the air | about 0.5 s | 0.50 s |

It trained about 60 times faster, with its own generated roads. But pilots trained in it didn't transfer. The best
reached 54% of the real Road 1 and never finished it. Varying the physics, adding hard practice courses and
swapping pilots between the sim and the real game didn't close the gap.

**Lesson:** every small difference (collisions, tile edges, timing) matters, and matching them by hand never ends.

## Attempt 4: An Open Source Port, Run Headless

Then I found [skyroads-sdl](https://github.com/pedrocatalao/skyroads-sdl), an open source port that runs the game's
own freeware data with a rewrite of its game logic. We cloned it unchanged and wrote a small C driver
(`ext/skyenv/sky_env.c`) that runs the game loop with no window. Once per frame it hands the ship's state to Python
and takes back the keys.

A full flight takes about 20 ms, against about 10 seconds in DOSBox. That's roughly 500 to 1000 times faster, with
the game's own physics and finish rule. We checked it against DOSBox (speed, jump height, lane positions), and a
side by side viewer flies the same pilot in both at once.

## Attempt 5: Training in the Port

**Setup:**

- 500 pilots. Each flies every unlocked road once, plus practice takeovers along each road and near the end.
- Fitness ranks each road and practice point separately, then takes 0.6 × the average + 0.4 × the worst road +
  0.25 × the practice average.
- Training starts on Road 1 only. The next road unlocks when half of the top 20 pilots finish every unlocked road,
  5 generations in a row.
- Roads already learned get a bonus for finishing them, so they don't get forgotten.

### What went wrong, and what fixed it

**1. Starting from the DOSBox pilots carried their bad habits over.** After 170 generations, Road 1 was finished in
only 1 to 4 out of 400 flights. Starting from scratch was faster: 83% of Road 1 within 40 seconds.

**2. Holding jump down.** About 88% of pilots held jump, which clears most of Road 1 but bounces over the final
tunnel. As I put it at the time, less is more, and sometimes just continuing with the momentum is good enough.
A tap-to-jump mode taught that, and that run mastered Road 1, including real finishes in DOSBox. But it also made
jumping hard to learn. Once I stopped caring about matching DOSBox timing exactly, going back to the game's own
held jump, exact timing and a decision every frame learned much faster.

**3. Forgetting.** Adding Road 2 wiped out Road 1. Finishes dropped from about 400 to about 10 over 38 generations.
A bonus for finishing old roads fixed it. Set too high (5000), it blocked pilots that were making progress on Road 2.
At 600, losing Road 1 completely costs more than anything Road 2 can give, but one missed finish can still be made up.

**4. Road 2 finishers that never spread.** From generation 774 to 1409, about 4 out of 500 pilots finished Road 2
every generation. That never grew, and at generation 1410 the last of them died out. Two reasons:

- With ranked fitness, finishing the road ranked one place above falling off at 99%.
- The practice pilot was the furthest one from the current generation only. When the finishers died out, so did the
  practice at the end of Road 2.

We rolled back to generation 1100, added a +400 bonus for finishing the newest road, and kept the furthest pilot
ever as the practice pilot. The finishers survived after that, but there were still only 4 or 5 of them.

**5. Mutation was breaking the skill.** To find out why, we took the 4 pilots that finished both roads, bred 60
offspring from each, and flew them:

| How the offspring were made | Still finish both roads |
|---|---|
| Default NEAT mutation (80% of weights changed) | 8% |
| Crossover with another pilot + default mutation | 2% |
| Gentle mutation (10% of weights, small changes) | 96% |
| Crossover with another pilot + gentle mutation | 29% |
| Crossover between two finishers + gentle mutation | 45% |

That explained it. Finishers survived, but almost none of their children kept the skill. Switching to gentle
mutation, with 75% of children bred from one parent, took Road 2 finishers from 6 to 79 in 5 generations.
Road 3 unlocked at generation 1265, and Road 2 finishers later reached about 470 out of 500.

**6. Everyone became the same pilot.** Gentle mutation made the population near copies of each other. All 500 hit
the same block early on Road 3 (row 10, about 12%) for 155 generations. 347 drifted right into it and 114 drifted left.
We added explorers: 20% of children get the old, strong mutation. One of them got past the block straight away.
But it had lost Roads 1 and 2 along the way, so it ranked near the bottom and died out.

The explorers also caused a side problem. Each one formed its own species, and the count jumped from 2 to 105.
NEAT never merges species, so we now re-sort everyone into species from scratch when the count gets too high.

**7. Half solutions were getting thrown away.** To keep explorers like that alive, we added protected seeds: the 5
pilots that get furthest on the newest road are kept every generation, even if they fail older roads, along with 25
gently mutated children. Road 3 went from 27% to 86% in 4 generations. The seeds also slowly won back Road 1 and
most of Road 2.

Then it hit a second wall at row 52 of 60. The left lane ends at row 49, and the only way on is to jump right onto
some short middle lanes, then jump again onto a strip on the far right. A pilot that makes the first jump and then
falls scores the same as one that rides straight off the end. So the half solution kept getting thrown away, and
nothing changed for 240 generations.

The fix was stepping stones (the idea behind MAP-Elites). Every flight records the last spot the ship stood on, by
row and lane. Each spot keeps the best pilot that got there, and the furthest spot in each lane gets planted back
every generation. A pilot that reaches the middle lanes and falls is now kept, and its children only need to learn the
second jump. Within 70 generations, the stepping stone in lane 2 moved from row 44 to row 49, and the right side of the
road was explored for the first time.

## Where It Stands

As of generation 1,870:

| Road | Status |
|---|---|
| Road 1 | Learned. About 270 to 490 of 500 pilots finish it every generation |
| Road 2 | Learned. Up to about 470 of 500 finish it, and the best pilot always does |
| Road 3 | The best all-round pilot stops at row 10 (12%). The furthest pilot reaches row 52 of 60 (87%), and the stepping stones are closing in on the crossing |

A generation of 500 pilots, each flying 21 times, takes about 30 seconds.

## Conclusion

Comparing the five attempts, the environment made the biggest difference. Real time DOSBox gave about 40 noisy
generations an hour. The headless port gives exact, repeatable flights about 1000 times faster, so fixes that took hours
to test took minutes.

After that, almost every problem was in the evolution itself, not the pilot.

1. **Get the exact game state and a correct finish signal first.** Screen reading and guessed signals gave false wins
   and hid the real problems.
2. **Don't hand-build a simulator of a game you can run.** The matched sim hit every measured number and still didn't
   transfer. Running the game's own logic did.
3. **If every algorithm stalls at the same score, fix the inputs.** GA, ES and NEAT all got stuck bunny hopping, and
   clear-path inputs got them out.
4. **Ranked fitness hides big jumps.** Finishing a road needs its own bonus, or it's only worth one rank.
5. **Test mutation directly.** Mutating a skilled pilot and flying the children explained hundreds of stuck generations
   in a few minutes.
6. **Keep the skill and keep exploring, separately.** Gentle mutation keeps skills but kills variety. Strong mutation
   explores but breaks skills. Doing both, plus species control that can merge species, keeps both.
7. **Distance alone is misleading at multi-step obstacles.** Protected seeds and stepping stones keep half solutions
   alive long enough to finish them.
8. **Practice where the population gets stuck.** Start practice from the furthest pilot ever, and leave enough room
   before the obstacle to set up for it.
9. **Keep checkpoints.** Every fix reset some of what evolution had settled into, and being able to roll back saved
   the run more than once.

### What's next

- Let the stepping stones get through Road 3, and see if that line wins Roads 1 and 2 back and unlocks Road 4.
- Breed seeds with the main population's finishers, since two finishers crossed together kept the skill 45% of the time.
- Try separate pilots for each road as rescue teams when the main run gets stuck.
- Turn DOSBox timing back on once roads are learned, so the pilots work on the real game again.
- Try novelty search for the hard spots, instead of hand-picking what counts as a stepping stone.

### Limitations

- One pilot has to keep every earlier road while learning the next one. The bonuses and seeds that make this work were
  tuned by hand.
- Training in the port no longer matches DOSBox timing, so results on the real game are mixed.
- The inputs, rewards, practice starts and stepping stones are my design choices. The pilots still only learn through
  selection and mutation.
- Roads 4 to 30 haven't been tried. Road 20 has gravity so high that jumping is impossible, and fuel and oxygen
  limits haven't been mapped yet.

## Appendix: Repo Layout

| Path | What it is |
|---|---|
| `starlane/` | The Starlane simulator, the shared inputs (`realobs.py`), the matched sim, and the port wrapper (`portenv.py`) |
| `bridge/` | DOSBox window control, screen capture, memory reading and live flying |
| `ext/skyenv/` | The headless driver for the skyroads-sdl port |
| `train.py`, `train_neat.py` | Attempt 1 trainers |
| `train_real.py`, `train_realsim.py` | Attempts 2 and 3 trainers |
| `train_port.py`, `train_forever.ps1` | Attempt 5 trainer, and a script that restarts it if it crashes |
| `watch_side_by_side.py`, `check_transfer.py` | DOSBox vs port viewer, and an hourly check on the real game |

## Appendix: Current Settings

| Setting | Value |
|---|---|
| Population | 500 |
| Flights | 1 per road, exact timing, held jump, a decision every frame |
| Practice | 4 evenly spaced takeovers, 2 near the sticking point, and 2 near the end of the road |
| Fitness | Ranked per road and practice point: 0.6 × average + 0.4 × worst + 0.25 × practice |
| Bonus for old roads | 600 × share finished |
| Bonus for the newest road | 400 for finishing |
| Unlocking | Half of the top 20 finish every road, 5 generations in a row |
| Mutation | Gentle (10% of weights, small changes), with 20% explorers on the default settings |
| Breeding | 75% from a single parent |
| Species | Target 12, re-sorted when over 24 |
| Seeds | The furthest stepping stone in each lane, plus 25 gentle children |
