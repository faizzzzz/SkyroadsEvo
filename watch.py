"""Watch an evolved pilot, or fly yourself.

    python watch.py runs/neat_full/best.pkl --difficulty 0.5
    python watch.py --human               # arrows steer/throttle, space jumps

Keys: R restart, N next level, +/- difficulty, F fast-forward, Esc quit.
"""
import argparse

import numpy as np
import pygame

from starlane import config as C
from starlane.level import generate
from starlane.pilot import Pilot
from starlane.policy import to_actions
from starlane.render import Renderer, W, H
from starlane.sim import BatchSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("genome", nargs="?", help="pilot file: .npz (GA/ES) or .pkl (NEAT)")
    p.add_argument("--human", action="store_true")
    p.add_argument("--difficulty", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--noise", type=float, default=0.0)
    a = p.parse_args()
    if not a.human and not a.genome:
        p.error("give a genome file or --human")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Starlane")
    clock = pygame.time.Clock()
    rend = Renderer(screen)
    if not a.human:
        pilot = Pilot(a.genome)
        obs_mode = pilot.obs_mode
        print(pilot.describe())
    else:
        obs_mode = "full"

    seed, diff, fast = a.seed, a.difficulty, False
    wins = runs = 0

    def new_sim():
        return BatchSim(generate(seed, diff), 1, obs_mode, a.noise, np.random.default_rng(seed))

    sim, end_timer, out = new_sim(), 0, None
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    sim, end_timer = new_sim(), 0
                elif e.key == pygame.K_n:
                    seed += 1
                    sim, end_timer = new_sim(), 0
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    diff = min(1.0, diff + 0.1)
                    sim, end_timer = new_sim(), 0
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    diff = max(0.0, diff - 0.1)
                    sim, end_timer = new_sim(), 0
                elif e.key == pygame.K_f:
                    fast = not fast

        for _ in range(4 if fast else 1):
            if not sim.alive[0]:
                break
            if a.human:
                k = pygame.key.get_pressed()
                steer = int(k[pygame.K_RIGHT]) - int(k[pygame.K_LEFT])
                thr = int(k[pygame.K_UP]) - int(k[pygame.K_DOWN])
                jump = bool(k[pygame.K_SPACE])
                sim.step(np.array([steer]), np.array([thr]), np.array([jump]))
            else:
                out = pilot.act(sim.observe())
                s, t, j = to_actions(out)
                sim.step(s, t, j)
                out = out[0]

        if not sim.alive[0]:
            if end_timer == 0:
                runs += 1
                wins += int(sim.reason[0] == C.WIN)
            end_timer += 1
            if end_timer > 60 and not a.human:   # auto-advance after 2 s
                seed += 1
                sim, end_timer = new_sim(), 0

        hud = {"level seed": seed, "difficulty": f"{diff:.1f}",
               "time": f"{sim.t:5.1f}s", "won": f"{wins}/{runs}",
               "pilot": "human" if a.human else obs_mode}
        rend.draw(sim, 0, hud, out)
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
