"""Live mushroom-body panel: watch the circuit think while the bird flies.

Draws, for each tick:
  - PN tuning responses (feature x bin dots, brightness = response)
  - both KC pools as grids (bright = the sparse coincidence code)
  - MBON approach/avoid drives (green/red dots, size = drive)
  - action values as bars (chosen action highlighted)
  - a rolling dopamine (RPE) trace
"""

from __future__ import annotations

from collections import deque

import numpy as np

PANEL_W = 480
BG = (24, 24, 32)
GRID_DIM = (40, 44, 52)
TEXT = (210, 210, 220)
ACTIVE = (255, 214, 64)
APPROACH = (80, 200, 120)
AVOID = (220, 90, 90)

FEATURES = ["height", "vy", "dist", "gap", "gap sz"]


class BrainPanel:
    """Renders a MushroomBody.observe() snapshot onto a pygame surface."""

    def __init__(self, num_features: int, num_bins: int, num_kc: int,
                 num_mbon: int, num_actions: int):
        import pygame

        self.pygame = pygame
        self.font = pygame.font.SysFont(None, 18)
        self.num_bins = num_bins
        self.pool = num_kc // num_actions
        self.num_mbon = num_mbon
        self.num_actions = num_actions
        self.feature_names = FEATURES[:num_features]
        self.rpe_trace: deque[float] = deque(maxlen=240)
        self.action_names = ["glide", "flap"]

    # -- helpers ----------------------------------------------------------

    def _dot(self, surf, x, y, r, color, intensity):
        if intensity <= 0.02:
            return
        c = tuple(int(v * min(1.0, intensity)) for v in color)
        self.pygame.draw.circle(surf, c, (int(x), int(y)), int(r))

    # -- sections -----------------------------------------------------------

    def _draw_pns(self, surf, pn, ox, oy):
        pygame = self.pygame
        label = self.font.render("projection neurons", True, TEXT)
        surf.blit(label, (ox, oy))
        cell = 22
        for f in range(pn.shape[0]):
            name = self.font.render(
                self.feature_names[f] if f < len(self.feature_names) else str(f),
                True, TEXT,
            )
            surf.blit(name, (ox, oy + 18 + f * cell + 6))
            for b in range(pn.shape[1]):
                self._dot(surf, ox + 30 + b * cell + 6, oy + 18 + f * cell + 8,
                          7, ACTIVE, float(pn[f, b]))

    def _draw_kc_pool(self, surf, kc_pool, ox, oy, title):
        pygame = self.pygame
        surf.blit(self.font.render(title, True, TEXT), (ox, oy))
        # lay the pool out as a dense grid
        cols = 24
        rows = int(np.ceil(len(kc_pool) / cols))
        cell = 8
        for i, on in enumerate(kc_pool):
            r, c = divmod(i, cols)
            x = ox + 6 + c * cell
            y = oy + 18 + r * cell
            if on:
                pygame.draw.rect(surf, ACTIVE, (x, y, 5, 5))
            else:
                pygame.draw.rect(surf, GRID_DIM, (x, y, 5, 5))

    def _draw_mbons(self, surf, app, avd, ox, oy):
        pygame = self.pygame
        surf.blit(self.font.render("MBONs", True, TEXT), (ox, oy))
        m = len(app)
        for i in range(m):
            x = ox + 6 + i * 20
            self._dot(surf, x, oy + 26, 7, APPROACH,
                      float(app[i]) / max(1e-6, np.max(app)) * 1.0)
            self._dot(surf, x, oy + 52, 7, AVOID,
                      float(avd[i]) / max(1e-6, np.max(avd)) * 1.0)
        a = self.font.render("approach", True, APPROACH)
        v = self.font.render("avoid", True, AVOID)
        surf.blit(a, (ox + 6, oy + 62))
        surf.blit(v, (ox + 6, oy + 78))

    def _draw_values(self, surf, q, chosen, ox, oy, width):
        pygame = self.pygame
        surf.blit(self.font.render("values", True, TEXT), (ox, oy))
        span = max(abs(float(q.min())), abs(float(q.max())), 0.05)
        for a in range(self.num_actions):
            y = oy + 20 + a * 26
            name = self.font.render(self.action_names[a], True, TEXT)
            surf.blit(name, (ox, y - 6))
            bx, bw = ox + 52, width - 60
            mid = bx + bw // 2
            pygame.draw.line(surf, GRID_DIM, (mid, y), (mid, y + 12), 1)
            v = float(q[a]) / span * (bw // 2)
            color = ACTIVE if a == chosen else (120, 120, 130)
            if v >= 0:
                pygame.draw.rect(surf, color, (mid, y, int(v), 12))
            else:
                pygame.draw.rect(surf, color, (mid + int(v), y, int(-v), 12))

    def _draw_rpe(self, surf, ox, oy, width, height):
        pygame = self.pygame
        surf.blit(self.font.render("dopamine (RPE)", True, TEXT), (ox, oy))
        mid = oy + height // 2 + 8
        pygame.draw.line(surf, GRID_DIM, (ox, mid), (ox + width, mid), 1)
        if not self.rpe_trace:
            return
        vals = np.array(self.rpe_trace)
        span = max(np.abs(vals).max(), 0.05)
        n = len(vals)
        pts = []
        for i, v in enumerate(vals):
            x = ox + int(i / max(1, self.rpe_trace.maxlen - 1) * (width - 8)) + 4
            y = mid - int(v / span * (height // 2 - 6))
            pts.append((x, y))
        if len(pts) > 1:
            pygame.draw.lines(surf, (120, 180, 255), False, pts, 1)

    # -- entry ------------------------------------------------------------

    def draw(self, surf, act: dict, chosen: int, rpe: float | None):
        """act: MushroomBody.observe() output for a single bird."""
        pygame = self.pygame
        pygame.draw.rect(surf, BG, (0, 0, PANEL_W, surf.get_height()))
        pn = act["pn"][0]      # (F, B)
        kc0 = act["kc"][0][0]  # pool 0 KC code
        kc1 = act["kc"][1][0]  # pool 1
        self._draw_pns(surf, pn, 12, 8)

        pool_grid_w = 24 * 8 + 10
        y_kc = 8 + 18 + pn.shape[0] * 22 + 6
        self._draw_kc_pool(surf, kc0[: self.pool], 12, y_kc, "Kenyon cells: glide pool")
        self._draw_kc_pool(surf, kc1[self.pool:], 12 + pool_grid_w + 10, y_kc,
                           "Kenyon cells: flap pool")

        y_mb = y_kc + 18 + int(np.ceil(self.pool / 24)) * 8 + 8
        self._draw_mbons(surf, act["app"][0][0], act["avd"][0][0], 12, y_mb)
        self._draw_mbons(surf, act["app"][1][0], act["avd"][1][0],
                         12 + pool_grid_w + 10, y_mb)

        y_q = y_mb + 100
        self._draw_values(surf, act["q"][0], chosen, 12, y_q, PANEL_W - 24)

        y_r = y_q + 78
        self._draw_rpe(surf, 12, y_r, PANEL_W - 24, 90)
        if rpe is not None:
            self.rpe_trace.append(float(rpe))
