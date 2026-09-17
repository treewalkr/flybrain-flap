"""Play the game yourself or watch the brain play, in realtime.

Human play (SPACE or click to flap):
    python -m flybrain_flap.game --human

Watch a trained brain play:
    python -m flybrain_flap.game --brain runs/flap/weights.npz

Keys: SPACE flap (human) / faster (brain), P pause, R restart, ESC quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .brain import MBConfig, MushroomBody
from .env import FlappyEnv

SCREEN_W, SCREEN_H = 480, 640
FPS = 60
SKY = (113, 197, 207)
GROUND = (222, 216, 149)
PIPE = (111, 196, 69)
PIPE_DARK = (86, 160, 55)
BIRD = (255, 204, 0)
BIRD_DARK = (200, 150, 0)
TEXT = (40, 40, 40)


class GameView:
    """Single-bird flappy game with pygame rendering."""

    def __init__(self, seed: int | None = None):
        import pygame

        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 32)
        self.big_font = pygame.font.SysFont(None, 64)
        self.env = FlappyEnv(1, seed=seed)
        self.obs = self.env.reset()
        self.paused = False

    # -- drawing ---------------------------------------------------------

    def _scale_y(self, world_y: float) -> int:
        """World y in [0, 1] (0 = floor) -> screen y."""
        return SCREEN_H - int(world_y * (SCREEN_H - 40)) - 20

    def draw(self, flash_flap: bool = False):
        pygame = self.pygame
        env = self.env
        world_to_screen = lambda wx: int(wx / 0.75 * SCREEN_W)
        bx = world_to_screen(env.BIRD_X)

        self.screen.fill(SKY)
        pygame.draw.rect(
            self.screen, GROUND, (0, self._scale_y(0.0), SCREEN_W, 40)
        )

        # next pipe
        col = world_to_screen(env.next_pipe_x[0])
        gap_top = self._scale_y(env.gap_center[0] + env.gap_half)
        gap_bot = self._scale_y(env.gap_center[0] - env.gap_half)
        width = 60
        pygame.draw.rect(self.screen, PIPE, (col, 0, width, gap_top))
        pygame.draw.rect(self.screen, PIPE_DARK, (col, 0, 8, gap_top))
        pygame.draw.rect(self.screen, PIPE, (col, gap_bot, width, SCREEN_H))
        pygame.draw.rect(self.screen, PIPE_DARK, (col, gap_bot, 8, SCREEN_H))
        pygame.draw.rect(
            self.screen, PIPE_DARK, (col - 4, gap_top - 16, width + 8, 16)
        )
        pygame.draw.rect(
            self.screen, PIPE_DARK, (col - 4, gap_bot, width + 8, 16)
        )

        # bird
        by = self._scale_y(env.y[0])
        r = 14 if flash_flap else 12
        pygame.draw.circle(self.screen, BIRD, (bx, by), r)
        pygame.draw.circle(self.screen, BIRD_DARK, (bx, by), r, 2)
        pygame.draw.circle(self.screen, TEXT, (bx + 5, by - 4), 2)

        # HUD
        score = self.font.render(
            f"pipes: {env.pipes_passed[0]}", True, TEXT
        )
        self.screen.blit(score, (12, 12))
        if self.paused:
            p = self.big_font.render("PAUSED", True, TEXT)
            self.screen.blit(
                p, (SCREEN_W // 2 - p.get_width() // 2, SCREEN_H // 2 - 32)
            )
        pygame.display.flip()

    def draw_game_over(self):
        pygame = self.pygame
        over = self.big_font.render("GAME OVER", True, TEXT)
        again = self.font.render(
            f"pipes: {self.env.pipes_passed[0]}  -  R restart, ESC quit",
            True, TEXT,
        )
        self.screen.blit(
            over, (SCREEN_W // 2 - over.get_width() // 2, SCREEN_H // 2 - 60)
        )
        self.screen.blit(
            again, (SCREEN_W // 2 - again.get_width() // 2, SCREEN_H // 2 + 10)
        )
        pygame.display.flip()

    # -- loop ------------------------------------------------------------

    def run(
        self,
        brain: MushroomBody | None = None,
        speed: int = 1,
        tps: int = 20,
    ):
        """Run the game loop. Human mode ticks the env `tps` times per
        second (flap inputs are buffered between ticks); brain mode takes
        `speed` env steps per 60 fps display frame."""
        pygame = self.pygame
        running = True
        game_over_frames = 0
        skip = max(1, round(FPS / tps))  # display frames per env tick
        frame = 0
        flap_buffered = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.obs = self.env.reset()
                        self.paused = False
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_SPACE:
                        if brain is None:
                            flap_buffered = True
                        elif speed < 8:
                            speed *= 2
                elif event.type == pygame.MOUSEBUTTONDOWN and brain is None:
                    flap_buffered = True

            if self.paused:
                self.clock.tick(FPS)
                self.draw()
                continue

            if brain is None:
                # human frame: step the env every `skip` display frames
                flapped_now = False
                done = False
                if frame % skip == 0:
                    actions = np.array(
                        [1 if flap_buffered else 0], dtype=np.int64
                    )
                    flap_buffered = False
                    flapped_now = bool(actions[0])
                    self.obs, _, done = self.env.step(actions)
                self.draw(flash_flap=flapped_now)
                if done:
                    self._wait_restart()
            else:
                # brain frames: `speed` env steps per display frame
                flash = False
                done = False
                for _ in range(speed):
                    actions, _, _ = brain.act(self.obs, epsilon=0.0)
                    flash = bool(actions[0] == 1)
                    self.obs, _, done = self.env.step(actions)
                    if done:
                        break
                self.draw(flash_flap=flash)
                if done:
                    game_over_frames += 1
                    if game_over_frames > 90:
                        game_over_frames = 0
                        self.obs = self.env.reset()
                    else:
                        self.draw_game_over()
                else:
                    game_over_frames = 0

            frame += 1
            self.clock.tick(FPS)
        pygame.quit()

    def _wait_restart(self, brain: MushroomBody | None = None):
        pygame = self.pygame
        self.draw_game_over()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        raise SystemExit
                    if event.key == pygame.K_r:
                        self.obs = self.env.reset()
                        return
            pygame.time.wait(30)
            self.draw_game_over()


def load_brain(path: str, obs_dim: int) -> MushroomBody:
    brain = MushroomBody(MBConfig(num_features=obs_dim))
    brain.load(path)
    return brain


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--human", action="store_true", help="play yourself")
    p.add_argument("--brain", type=str, help="weights.npz to watch")
    p.add_argument("--speed", type=int, default=1,
                   help="env steps per frame in brain mode")
    p.add_argument("--tps", type=int, default=60,
                   help="env ticks per second in human mode")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    view = GameView(seed=args.seed)
    if args.human:
        view.run(brain=None, tps=args.tps)
    elif args.brain:
        brain = load_brain(args.brain, view.env.obs().shape[1])
        view.run(brain=brain, speed=args.speed)
    else:
        p.error("choose --human or --brain WEIGHTS")


if __name__ == "__main__":
    main()
