"""Vectorized Flappy-Bird-like environment.

N birds fly through the same pipe sequence. Physics is a fixed tick;
the only action is flap (1) or do nothing (0).
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 5
NUM_ACTIONS = 2

FLAP = 1


class FlappyEnv:
    """N parallel birds, one shared pipe sequence.

    Rewards:
      +1.0 for passing a pipe
      +0.01 per step survived
      +0.05 * closeness to the gap center height (dense shaping)
      -1.0 on death
    An episode is done when the bird dies. The batch resets when all
    birds are dead.
    """

    # classic flappy proportions, in screen-height units per 60 Hz tick:
    # a flap lifts ~8% of the screen over a ~0.4 s arc, gravity takes ~1.2 s
    # to pull the bird from mid-screen to the floor
    GRAVITY = 0.0005
    FLAP_VELOCITY = 0.009
    MAX_FALL_SPEED = 0.02
    PIPE_SPEED = 0.005
    PIPE_SPACING = 0.42
    GAP_HALF = 0.15
    BIRD_X = 0.25
    WORLD_HEIGHT = 1.0

    def __init__(
        self,
        num_envs: int,
        seed: int | None = None,
        gap_half: float = GAP_HALF,
        shaping: float = 0.05,
    ):
        assert num_envs >= 1
        self.num_envs = num_envs
        self.rng = np.random.default_rng(seed)
        self.gap_half = gap_half
        self.shaping = shaping
        self.reset()

    def obs(self) -> np.ndarray:
        return self._obs()

    def reset(self) -> np.ndarray:
        n = self.num_envs
        self.y = np.full(n, 0.5)
        self.vy = np.zeros(n)
        self.alive = np.ones(n, dtype=bool)
        # first pipe starts off to the right, ~1.2 s away
        self.next_pipe_x = np.full(n, 0.6)
        self.gap_center = self.rng.uniform(
            self.gap_half + 0.05, self.WORLD_HEIGHT - self.gap_half - 0.05, n
        )
        self.steps = np.zeros(n, dtype=np.int64)
        self.pipes_passed = np.zeros(n, dtype=np.int64)
        return self._obs()

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        live = self.alive
        px = self.next_pipe_x[live]
        gc = self.gap_center[live]
        obs[live, 0] = self.y[live] * 2.0 - 1.0
        obs[live, 1] = self.vy[live] / self.MAX_FALL_SPEED  # ~[-1, 0.45]
        obs[live, 2] = (px - self.BIRD_X) / self.PIPE_SPACING
        obs[live, 3] = (gc - self.y[live]) * 2.0
        obs[live, 4] = self.gap_half * 4.0
        return obs

    def step(self, actions: np.ndarray):
        """actions: int array (N,) of 0/1. Returns obs, rewards, done."""
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs,)

        # flap sets upward velocity
        flapped = self.alive & (actions == FLAP)
        self.vy[flapped] = self.FLAP_VELOCITY

        # physics for live birds
        self.vy[self.alive] -= self.GRAVITY
        np.clip(self.vy, -self.MAX_FALL_SPEED, None, out=self.vy)
        self.y[self.alive] += self.vy[self.alive]
        self.next_pipe_x[self.alive] -= self.PIPE_SPEED

        # deaths: out of bounds or pipe collision
        hit_pipe = (np.abs(self.y - self.gap_center) > self.gap_half) & (
            np.abs(self.next_pipe_x - self.BIRD_X) < 0.03
        )
        dead = self.alive & (
            (self.y <= 0.0) | (self.y >= self.WORLD_HEIGHT) | hit_pipe
        )
        rewards = np.full(self.num_envs, 0.01, dtype=np.float32)
        # dense shaping: reward being near the upcoming gap's height
        align = 1.0 - np.minimum(
            np.abs(self.y - self.gap_center) / self.gap_half, 1.0
        )
        rewards += (self.shaping * align).astype(np.float32)
        rewards[dead] = -1.0
        self.alive &= ~dead

        # pipe passed once it is fully behind the bird
        passed = self.alive & (self.next_pipe_x < self.BIRD_X - 0.03)
        rewards[passed] += 1.0
        self.pipes_passed += passed
        new_gc = self.rng.uniform(
            self.gap_half + 0.05, self.WORLD_HEIGHT - self.gap_half - 0.05,
            self.num_envs,
        )
        self.gap_center = np.where(passed, new_gc, self.gap_center)
        self.next_pipe_x = np.where(
            passed, self.next_pipe_x + self.PIPE_SPACING, self.next_pipe_x
        )

        self.steps += self.alive
        rewards[~self.alive & ~dead] = 0.0  # birds already dead before this step
        done = bool(np.all(~self.alive))
        return self._obs(), rewards, done
