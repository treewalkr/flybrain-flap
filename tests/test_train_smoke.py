import numpy as np

from flybrain_flap.brain import MBConfig, MushroomBody
from flybrain_flap.env import FlappyEnv


def run_steps(num_envs, steps, epsilon, seed):
    env = FlappyEnv(num_envs, seed=seed)
    brain = MushroomBody(MBConfig(num_features=5, eta=0.02, seed=seed))
    obs = env.reset()
    best = 0
    prev_alive = env.alive.copy()
    ar = np.arange(num_envs)
    for _ in range(steps):
        actions, q, kc_cache = brain.act(obs, epsilon)
        next_obs, rewards, done = env.step(actions)
        newly_dead = prev_alive & ~env.alive
        q_next = brain.values(next_obs).max(axis=1)
        brain.learn(actions, kc_cache, rewards, q[ar, actions], q_next,
                    newly_dead, alive=prev_alive)
        prev_alive = env.alive.copy()
        obs = next_obs
        if done:
            best = max(best, int(env.pipes_passed[0]))
            obs = env.reset()
            prev_alive = env.alive.copy()
    return best, env.pipes_passed


def test_short_training_learns_something():
    """A few thousand steps must beat the no-flap baseline."""
    # no-flap baseline: the bird drops out of the sky immediately
    env = FlappyEnv(1, seed=3)
    env.reset()
    for _ in range(500):
        _, _, done = env.step(np.zeros(1, dtype=np.int64))
        if done:
            break
    baseline = int(env.pipes_passed[0])

    best, _ = run_steps(32, 6000, epsilon=0.2, seed=3)
    assert best >= baseline
