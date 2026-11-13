# Teaching a Neural Network to Play Skyroads with Evolution

Skyroads is a DOS game from 1993. You fly a ship down a road floating in space, and you steer, speed up, slow
down and jump over holes and blocks. Every road ends in a tunnel, and you have to drive into it to finish. I
wanted to see if an evolutionary algorithm could learn to play it, the same way SethBling's MarIO learned to play
Super Mario World. There is no gradient descent and no training on human play. The pilots that fly the furthest
get to breed, and mutation and crossover take care of the rest.

I worked on this over four days, from 10 to 13 November 2026. Claude Code wrote most of the code while I directed
it, tested it and decided what to try next. This write-up goes through the approach, what went wrong, what fixed
it and where it stands now.

## Goal

The goal was to evolve one pilot that can finish Roads 1 to 3, and ideally all 30 roads. I set a few rules at the
start. Learning had to happen through evolution only, so selection, mutation and crossover. Shaping the task was
allowed, like adding inputs, practice starts and reward terms, but hand-coding the behaviour was not. The game would
run from my own copy without modifying it. Skyroads is freeware, but none of its files are in this repo. I also
decided not to recreate the real levels inside a custom simulator.

## The Pilot

Each pilot is a NEAT network, using the neat-python library. A network starts very small, with a few random links
from the inputs to the outputs and no hidden neurons. NEAT adds links and neurons over time as they help.

The pilot gets 165 inputs. Most of them are a grid of the road ahead: 13 lanes centred on the ship by 11 distance
bands, covering every row up to 20 rows ahead. Each cell is a hole, road, tunnel or block. On top of that, the pilot
gets how far ahead each lane stays clear, its position and speed, its height above the road, the road's gravity, and
how close the end of the road is.

There are 3 outputs: steer, throttle and jump. Steering is proportional, so the size of the output decides how often
the key is held down. This lets the pilot make small corrections on narrow parts of the road.

Each flight is scored as 1000 × progress + 500 × end approach + 1000 × finished. Progress only counts while the ship
is at road level, and it stops at 99% unless the road is actually finished. The end approach score is the share of
the last 5 rows that the ship spends on the ground in the tunnel's lane.

## Starting with a Simulator

I started with a simulator of our own called Starlane. It has similar mechanics to Skyroads, with 7 lanes, holes,
blocks, special tiles, fuel and oxygen, but its levels are randomly generated. A generation of 200 pilots flying 16
levels took about 3.5 seconds.

For a base case, I evolved a fixed network with a genetic algorithm and with an evolution strategy. I tested the
results on 40 easy levels that the pilots had never seen:

- Genetic algorithm - 12/40
- Evolution strategy, step 0.05 - 7/40
- Evolution strategy, step 0.02 - 8/40

Next, I switched to NEAT, which won 18 to 19 out of 40 with networks of about 19 links instead of 10,000 weights. It
was better, but every method stopped improving at about the same point. When I replayed the crashes, I found that the
pilots had learned to hold full throttle and jump all the time. Hopping clears holes and low blocks without any
effort, so it scores well, but it does not get the ship through walls. To steer into a gap, the network has to compare
its left and right side, and small mutations rarely find that.

The fix was to give the pilots better inputs. I added a clear path input for each lane, which says how far ahead that
lane stays clear. With these inputs, the best pilot went from 19/40 to 30/40, and it was the first pilot to move up the
difficulty levels. Since all three algorithms got stuck at the same score, the problem was what the pilots could see,
not the algorithm.

## Moving to the Real Game

When I tried the simulator pilot on the real game in DOSBox, it lasted about 6 seconds on Road 1. The physics were
different, reading the road from screen colours only worked on one road, and a ship inside a tunnel looked the same as
a crash. So I moved the training onto the real game itself, running 6, then 12, then 18 DOSBox windows at the same time.

MarIO read the game's memory to get its inputs, so I did the same. The trainer reads the running game's memory, without
writing to it, to get the ship's position, height, speed and the road cells ahead. This gave exact falls, crashes and
finishes. While watching it, I noticed that every road ends in a tunnel, and the ship has to drive into it at road level
to finish. Pilots flying off the end of the road had been counted as wins. Entering the tunnel sets the ship's height
straight to 0, while a fall sinks slowly, so that became the finish signal. I also noticed that some roads have different
gravity, so each road now gets one standing jump to measure it, and gravity became an input.

The hopping habit came along from the simulator. In the replays, the jump key was held down the whole way over the final
tunnel. A small penalty for jumping, along with the end approach reward, brought the time spent jumping down from 88% to
about 13%. I later removed the penalty, because jumping is fine everywhere except the end of the road.

The next problem was that pilots did not reach the end often enough to learn it. To fix this, I added practice flights.
The best pilot flies the road, and each new pilot takes over 12 rows before the end. Within 5 generations, 4 out of 40
pilots were finishing the road for real. The first pilot to finish a whole road on its own came at generation 107.
Roads 2 and 3 were still stuck at 18% and 25%, so I moved the takeover point to just before wherever the furthest pilot
got stuck. That way, practice always happens at the spot the population needs to learn next.

There were other issues along the way. The lane mapping was off by almost half a lane, so pilots were wrong about which
lane they were in about half the time. After fixing it, tunnel entries went from about 45 to about 250 per generation.
Results were also noisy, and the same pilot could finish one flight and crash the next. Syncing the pilot's decisions to
the game's own updates helped, but real time play stayed noisy, and with only 40 pilots, selection was partly luck.

It was also slow. Faster screen capture, one process per window, DOSBox fast forward and shorter menu waits brought a
generation from about 240 seconds down to about 85, which is still only about 42 generations an hour. Road 1 was
finished many times. Road 2 was finished once in a practice flight, and it showed up as completed in the game's road
menu. Neither was reliable.

## A Simulator Matched to the Game

With access to the game's memory, I could measure the real physics and build a simulator that matched them. Top speed
was 6.03 rows per second in both, it took 4.10 seconds to reach top speed in the game and 4.07 in the simulator, and the
jump height and time in the air matched as well. This simulator used its own generated roads and trained about 60 times
faster.

However, the pilots trained in it did not do well on the real game. The best one reached 54% of Road 1 and never
finished it. Varying the physics, adding harder practice courses and swapping pilots between the simulator and the real
game did not close the gap. Small differences in collisions, tile edges and timing all mattered, and matching them by
hand did not have an end.

## Using an Open Source Port

Eventually, I found skyroads-sdl (https://github.com/pedrocatalao/skyroads-sdl), an open source port that runs the
game's own freeware data with a rewrite of its game logic. I cloned it without changes and wrote a small C driver,
`ext/skyenv/sky_env.c`, that runs the game loop without a window. Once per frame, it sends the ship's state to Python and
gets back the keys to press.

A full flight takes about 20 milliseconds, compared to about 10 seconds in DOSBox. That is roughly 500 to 1000 times
faster, with the game's own physics and finish rule. I checked the port against DOSBox for speed, jump height and lane
positions, and I set up a viewer that flies the same pilot in both games side by side.

## Training in the Port

In the port, each generation has 500 pilots. Each pilot flies every unlocked road once, along with practice flights
that start along each road and near its end. Fitness ranks each road and practice point separately, and then combines
them as 0.6 × the average + 0.4 × the worst road + 0.25 × the practice average. Training starts on Road 1 only. The next
road unlocks when half of the top 20 pilots finish every unlocked road for 5 generations in a row, and roads that are
already learned give a bonus for finishing them.

At first, I started the port training from the pilots trained in DOSBox. After 170 generations, Road 1 was only finished
in 1 to 4 out of 400 flights, so I restarted from scratch. The new population reached 83% of Road 1 within 40 seconds.

The next problem was jumping. About 88% of pilots held the jump key down, which gets through most of Road 1 but bounces
over the final tunnel. Less is more, and sometimes just continuing with the momentum is good enough. A tap to jump mode
taught them that, and that run learned Road 1, including real finishes in DOSBox. However, it also made jumping itself
hard to learn. Once I stopped caring about matching the DOSBox timing exactly, I went back to the game's own held jump,
exact timing and a decision every frame, and the pilots learned much faster.

Adding Road 2 caused the pilots to forget Road 1, and finishes dropped from about 400 to about 10 over 38 generations. A
bonus for finishing old roads fixed this. When the bonus was 5000, it blocked pilots that were making progress on Road 2.
At 600, losing Road 1 completely costs more than anything Road 2 can give, but one missed finish can still be made up.

From generation 774 to 1409, about 4 out of 500 pilots finished Road 2 every generation, but that number never grew,
and at generation 1410 the last of them died out. There were two reasons for this. With ranked fitness, finishing the
road only ranked one place above falling off at 99%. Also, the practice flights were carried by the furthest pilot of the
current generation, so when the finishers died out, the practice at the end of Road 2 went away with them. I rolled back
to generation 1100, added a bonus of 400 for finishing the newest road, and kept the furthest pilot ever for practice.
After that, the finishers survived, but there were still only 4 or 5 of them.

To find out why, I took the 4 pilots that finished both roads, bred 60 children from each, and flew them. The results
were:

- Default NEAT mutation, which changes 80% of the weights - 8% still finished both roads
- Crossover with another pilot and default mutation - 2%
- Gentle mutation, which changes 10% of the weights by small amounts - 96%
- Crossover with another pilot and gentle mutation - 29%
- Crossover between two finishers and gentle mutation - 45%

As observed in the results above, the finishers were surviving, but almost none of their children kept the skill. I
switched to gentle mutation and bred 75% of the children from a single parent. Road 2 finishers went from 6 to 79 within
5 generations, and Road 3 unlocked at generation 1265. Later, about 470 out of 500 pilots were finishing Road 2.

Gentle mutation also made the pilots almost identical to each other. All 500 of them hit the same block early on Road 3,
at row 10 or about 12%, for 155 generations. To bring back variety, I added explorers, which are 20% of the children
mutated with the old, stronger settings. One of them got past the block right away, but it had lost Roads 1 and 2 along
the way, so it ranked near the bottom and died out. The explorers also created a lot of species, from 2 to 105, and NEAT
does not merge species on its own, so the trainer now sorts everyone into species again when the count gets too high.

To keep pilots like that explorer alive, I added protected seeds. The 5 pilots that get the furthest on the newest road
are kept every generation, even if they fail the older roads, along with 25 children bred from them with gentle
mutation. Road 3 went from 27% to 86% in 4 generations, and the seeds slowly won back Road 1 and most of Road 2.

Then the pilots hit another wall at row 52 of 60. The left lane ends at row 49, and the only way forward is to jump right
onto a few short lanes in the middle, and then jump again onto a strip on the far right. A pilot that makes the first jump
and then falls scores the same as a pilot that rides straight off the end, so the pilots that were halfway there kept
getting thrown out. Nothing changed for 240 generations.

To fix this, I used stepping stones, which is the idea behind MAP-Elites. Every flight records the last spot the ship
stood on, by row and lane. Each spot keeps the best pilot that got there, and the furthest spot in each lane is planted
back into the population every generation. A pilot that reaches the middle lanes and then falls is now kept, and its
children only have to learn the second jump. Within 70 generations, the stepping stone in lane 2 moved from row 44 to row
49, and pilots reached the right side of the road for the first time.

## Where It Stands

At generation 1,870, Roads 1 and 2 are learned. About 270 to 490 out of 500 pilots finish Road 1 every generation, up to
about 470 finish Road 2, and the best pilot always finishes both. On Road 3, the best overall pilot still stops at row 10,
but the furthest pilot reaches row 52 out of 60, and the stepping stones are getting closer to the crossing. A
generation of 500 pilots, each flying 21 times, takes about 30 seconds.

## Conclusion

To summarize, the environment made the biggest difference. DOSBox in real time gave about 40 noisy generations an hour,
while the port gives exact, repeatable flights about 1000 times faster. Fixes that took hours to test in DOSBox took
minutes in the port. After that, almost every problem was in how the pilots were bred and scored, not in the pilots
themselves.

Some lessons from this project:

- Get the exact game state and a correct finish signal first. Reading the screen and guessing gave false wins and hid
  the real problems.
- Do not build your own simulator of a game you can already run. The matched simulator hit every number I measured and
  still did not work on the real game.
- If every algorithm gets stuck at the same score, look at the inputs.
- Ranked fitness hides big jumps like finishing a road, so finishing needs its own bonus.
- Test mutation directly. Breeding children from a good pilot and flying them explained hundreds of stuck generations in
  a few minutes.
- Keep a balance between keeping skills and exploring. Gentle mutation keeps skills but makes every pilot the same, and
  strong mutation explores but breaks skills.
- Scoring by distance alone does not work well when an obstacle takes more than one move. Protected seeds and stepping
  stones keep the pilots that are halfway there.
- Start practice where the population gets stuck, and leave enough room before the obstacle.
- Save checkpoints often. Being able to roll back saved the run more than once.

Next, I want to let the stepping stones get through Road 3 and see if they can win back Roads 1 and 2 and unlock Road 4.
I also want to try breeding the seeds with the pilots that already finish the old roads, since two finishers bred
together kept the skill 45% of the time. Other ideas are separate pilots for each road that can help when the main run
gets stuck, turning the DOSBox timing back on once the roads are learned, and novelty search for the hardest spots.

There are some limitations. One pilot has to remember every earlier road while it learns the next one, and the bonuses
and seeds that make this work were tuned by hand. Training in the port no longer matches the DOSBox timing, so results
on the real game are mixed. Roads 4 to 30 have not been tried yet. Road 20 has so much gravity that jumping is not
possible, and fuel and oxygen have not been mapped.

## Repo Layout

- `starlane/` - the Starlane simulator, the shared inputs (`realobs.py`), the matched simulator, and the port wrapper
  (`portenv.py`)
- `bridge/` - DOSBox window control, screen capture, memory reading and live flying
- `ext/skyenv/` - the C driver that runs the skyroads-sdl port without a window
- `train.py`, `train_neat.py` - training in Starlane
- `train_real.py`, `train_realsim.py` - training in DOSBox and in the matched simulator
- `train_port.py`, `train_forever.ps1` - training in the port, and a script that restarts it if it crashes
- `watch_side_by_side.py`, `check_transfer.py` - the side by side viewer, and an hourly check on the real game
