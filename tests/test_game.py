"""Headless smoke test for the pygame viewer (SDL dummy driver)."""

import os

import pytest

pygame = pytest.importorskip("pygame")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from flybrain_flap.brain import MBConfig, MushroomBody  # noqa: E402
from flybrain_flap.game import GameView, load_brain  # noqa: E402


def test_brain_mode_renders_frames(tmp_path):
    brain = MushroomBody(MBConfig(num_features=5, seed=0))
    weights = str(tmp_path / "w.npz")
    brain.save(weights)

    view = GameView(seed=0)
    loaded = load_brain(weights, view.env.obs().shape[1])

    frames = 0
    done = False
    while frames < 200 and not done:
        for _ in pygame.event.get():
            pass
        actions, _, _ = loaded.act(view.obs, epsilon=0.0)
        view.obs, _, done = view.env.step(actions)
        view.draw(flash_flap=bool(actions[0]))
        frames += 1
    view.pygame.quit()
    assert frames >= 2  # at least a couple of frames rendered before death


def test_human_step_advances_env():
    import numpy as np

    view = GameView(seed=0)
    y0 = view.env.y[0]
    view.obs, _, _ = view.env.step(np.array([1], dtype=np.int64))
    view.draw(flash_flap=True)
    view.pygame.quit()
    assert view.env.y[0] > y0  # flapping moves the bird up
