import numpy as np

from flybrain_flap.env import FlappyEnv


def test_reset_shapes():
    env = FlappyEnv(8, seed=0)
    obs = env.reset()
    assert obs.shape == (8, 5)
    assert np.all(env.alive)


def test_step_shapes_and_bounds():
    env = FlappyEnv(4, seed=0)
    obs = env.reset()
    actions = np.zeros(4, dtype=np.int64)
    for _ in range(500):
        obs, rewards, done = env.step(actions)
        assert obs.shape == (4, 5)
        assert rewards.shape == (4,)
    # doing nothing forever must kill every bird
    assert done
    assert not env.alive.any()


def test_death_and_reset():
    env = FlappyEnv(2, seed=0)
    env.reset()
    actions = np.zeros(2, dtype=np.int64)
    done = False
    for _ in range(2000):
        _, _, done = env.step(actions)
        if done:
            break
    assert done
    obs = env.reset()
    assert env.alive.all()
    assert obs.shape == (2, 5)


def test_flap_counteracts_gravity():
    env = FlappyEnv(1, seed=0)
    env.reset()
    flap = np.ones(1, dtype=np.int64)
    y0 = env.y[0]
    for _ in range(10):
        env.step(flap)
    assert env.y[0] > y0
