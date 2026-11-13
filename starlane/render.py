"""Pseudo-3D chase-camera renderer for Starlane (pygame)."""
import numpy as np
import pygame

from . import config as C
from .level import PAD_COLS

W, H = 960, 600
HORIZON = 0.36 * H
FOCAL = 520.0
CAM_BACK, CAM_UP = 4.5, 2.4
VIEW_ROWS = 45

TILE = {
    C.K_NORMAL: ((52, 64, 92), (60, 74, 104)),
    C.K_BOOST: ((40, 170, 120), (50, 190, 135)),
    C.K_STICKY: ((120, 70, 150), (132, 80, 162)),
    C.K_BURN: ((215, 80, 40), (235, 100, 50)),
    C.K_FUEL: ((60, 130, 220), (75, 145, 235)),
}
BLOCK_TOP = (110, 125, 160)
TALL_TOP = (150, 160, 190)


def _shade(c, f):
    return tuple(int(min(255, v * f)) for v in c)


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("consolas", 16)
        self.big = pygame.font.SysFont("consolas", 32, bold=True)
        rng = np.random.default_rng(7)
        self.stars = [(int(rng.integers(0, W)), int(rng.integers(0, H)), int(rng.integers(90, 255)))
                      for _ in range(160)]
        self.bg = pygame.Surface((W, H))
        for yy in range(H):
            f = yy / H
            self.bg.fill((int(6 + 14 * f), int(4 + 8 * f), int(18 + 30 * f)), (0, yy, W, 1))

    def _proj(self, X, Y, Z, cam):
        cx, cy, cz = cam
        dz = Z - cz
        if dz < 0.3:
            return None
        return (W / 2 + (X - cx) * FOCAL / dz, HORIZON - (Y - cy) * FOCAL / dz)

    def _poly(self, pts, color, cam):
        p = [self._proj(*q, cam) for q in pts]
        if any(v is None for v in p):
            return
        pygame.draw.polygon(self.screen, color, p)

    def draw(self, sim, i=0, hud: dict | None = None, outputs=None):
        s = self.screen
        s.blit(self.bg, (0, 0))
        for x, y, b in self.stars:
            s.set_at((x, y), (b, b, b))
        lv = sim.level
        sx, sy, sz = float(sim.x[i]), float(sim.y[i]), float(sim.z[i])
        cam = (C.LANES / 2 + (sx - C.LANES / 2) * 0.6, max(sy, 0) + CAM_UP, sz - CAM_BACK)
        r0 = max(int(cam[2]) - 1, 0)
        r1 = min(int(sz) + VIEW_ROWS, lv.top.shape[0] - 1)
        ship_drawn = False
        for r in range(r1, r0 - 1, -1):
            if not ship_drawn and r < np.floor(sz - C.HALF_L):
                self._ship(sim, i, cam)
                ship_drawn = True
            fade = max(0.25, 1.0 - (r - sz) / VIEW_ROWS)
            # draw lanes outside-in so side faces overlap correctly
            order = sorted(range(C.LANES), key=lambda c: -abs(c + 0.5 - cam[0]))
            for c in order:
                top = float(lv.top[r, c + PAD_COLS])
                if top <= C.GAP + 1:
                    continue
                k = int(lv.kind[r, c + PAD_COLS])
                if r == lv.length or r == lv.length + 1:
                    col = (200, 200, 220) if (c + r) % 2 else (40, 40, 60)
                elif top > 0:
                    col = TALL_TOP if top > C.BLOCK_H + 0.01 else BLOCK_TOP
                else:
                    col = TILE[k][(r + c) % 2]
                col = _shade(col, fade)
                x0, x1, z0, z1 = c + 0.04, c + 0.96, r, r + 1
                base = min(top, 0.0) - 0.25 if top <= 0 else 0.0
                # side face (visible from camera side)
                if cam[0] < x0:
                    self._poly([(x0, top, z0), (x0, top, z1), (x0, base, z1), (x0, base, z0)],
                               _shade(col, 0.55), cam)
                elif cam[0] > x1:
                    self._poly([(x1, top, z0), (x1, top, z1), (x1, base, z1), (x1, base, z0)],
                               _shade(col, 0.55), cam)
                # front face
                if cam[1] > base:
                    self._poly([(x0, top, z0), (x1, top, z0), (x1, base, z0), (x0, base, z0)],
                               _shade(col, 0.7), cam)
                # top face
                self._poly([(x0, top, z0), (x1, top, z0), (x1, top, z1), (x0, top, z1)], col, cam)
        if not ship_drawn:
            self._ship(sim, i, cam)
        self._hud(sim, i, hud or {}, outputs)

    def _ship(self, sim, i, cam):
        x, y, z = float(sim.x[i]), float(sim.y[i]), float(sim.z[i])
        if not sim.reason[i] in (C.RUNNING, C.WIN):
            col = (255, 120, 60)
        else:
            col = (235, 235, 245)
        sup = float(sim.top_at(np.array([z]), np.array([x]))[0])
        if sup > C.GAP + 1 and sup <= y + 0.01:  # shadow
            self._poly([(x - 0.3, sup + 0.01, z - 0.4), (x + 0.3, sup + 0.01, z - 0.4),
                        (x + 0.3, sup + 0.01, z + 0.4), (x - 0.3, sup + 0.01, z + 0.4)], (15, 15, 25), cam)
        hw, hl, h = C.HALF_W, C.HALF_L, 0.28
        nose = (x, y + 0.12, z + hl + 0.15)
        bl, br = (x - hw, y + 0.02, z - hl), (x + hw, y + 0.02, z - hl)
        tl, tr = (x - hw * 0.5, y + h, z - hl), (x + hw * 0.5, y + h, z - hl)
        self._poly([bl, br, tr, tl], _shade(col, 0.6), cam)          # back
        self._poly([tl, tr, nose], col, cam)                          # canopy
        self._poly([bl, tl, nose], _shade(col, 0.8), cam)
        self._poly([br, tr, nose], _shade(col, 0.8), cam)
        if sim.vz[i] > 1:  # engine glow
            p = self._proj(x, y + 0.12, z - hl - 0.02, cam)
            if p:
                pygame.draw.circle(self.screen, (255, 170, 60), (int(p[0]), int(p[1])),
                                   max(2, int(3 + sim.vz[i] / 3)))

    def _bar(self, x, y, label, v, col):
        s = self.screen
        s.blit(self.font.render(label, True, (200, 200, 220)), (x, y))
        pygame.draw.rect(s, (40, 40, 60), (x + 60, y + 3, 120, 12))
        pygame.draw.rect(s, col, (x + 60, y + 3, int(120 * max(0.0, min(1.0, v))), 12))

    def _hud(self, sim, i, hud, outputs):
        s = self.screen
        self._bar(12, 10, "SPEED", sim.vz[i] / C.V_MAX, (240, 200, 80))
        self._bar(12, 30, "FUEL", sim.fuel[i], (80, 160, 240))
        self._bar(12, 50, "OXYGEN", sim.oxygen[i], (120, 220, 160))
        self._bar(12, 70, "ROAD", sim.z[i] / sim.level.length, (200, 200, 220))
        y = 10
        for k, v in hud.items():
            s.blit(self.font.render(f"{k}: {v}", True, (200, 200, 220)), (W - 260, y))
            y += 20
        if outputs is not None:
            for j, name in enumerate(("steer", "throttle", "jump")):
                v = float(outputs[j])
                bx, by = W - 260, H - 70 + j * 20
                s.blit(self.font.render(name, True, (200, 200, 220)), (bx, by))
                pygame.draw.rect(s, (40, 40, 60), (bx + 90, by + 3, 140, 12))
                mid = bx + 160
                pygame.draw.rect(s, (230, 120, 200), (min(mid, mid + v * 70), by + 3, abs(v) * 70, 12))
        if sim.reason[i] != C.RUNNING:
            msg = C.REASONS[sim.reason[i]].upper()
            t = self.big.render(msg, True, (255, 230, 120))
            s.blit(t, (W / 2 - t.get_width() / 2, H * 0.18))
