"""Watch a trained policy play, rendered as ASCII frames.

Example:
    python -m flybrain_flap.play runs/flap/weights.npz --episodes 3
"""

from __future__ import annotations

import argparse

import numpy as np

from .brain import MBConfig, MushroomBody
from .env import FlappyEnv

WIDTH = 40
HEIGHT = 20


def render(env: FlappyEnv, i: int) -> str:
    grid = [[" "] * WIDTH for _ in range(HEIGHT)]

    # next pipe
    px = env.next_pipe_x[i]
    gc = env.gap_center[i]
    col = int(px * WIDTH)
    if 0 <= col < WIDTH:
        top = int((gc - env.gap_half) * HEIGHT)
        bot = int((gc + env.gap_half) * HEIGHT)
        for r in range(HEIGHT):
            if r < top or r > bot:
                grid[r][col] = "|"

    # bird
    row = HEIGHT - 1 - int(env.y[i] * HEIGHT)
    if 0 <= row < HEIGHT:
        grid[row][int(env.BIRD_X * WIDTH)] = "O"

    return "\n".join("".join(row) for row in grid)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("weights")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--sleep", type=float, default=0.05)
    args = p.parse_args()

    env = FlappyEnv(1, seed=42)
    brain = MushroomBody(MBConfig(num_features=env.obs().shape[1]))
    brain.load(args.weights)

    for ep in range(args.episodes):
        obs = env.reset()
        steps = 0
        while True:
            actions, _, _ = brain.act(obs, epsilon=0.0)
            obs, _, done = env.step(actions)
            steps += 1
            print(f"\n== episode {ep + 1}  pipes {env.pipes_passed[0]} ==")
            print(render(env, 0))
            if done or steps > 10000:
                break
        print(f"episode {ep + 1}: pipes passed = {env.pipes_passed[0]}")


if __name__ == "__main__":
    main()
