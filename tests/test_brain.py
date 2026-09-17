import numpy as np

from flybrain_flap.brain import MBConfig, MushroomBody


def make_brain():
    return MushroomBody(MBConfig(num_features=5, num_kc=64, kc_active=8, seed=0))


def rollout(brain, obs, epsilon):
    actions, q, kc_cache = brain.act(obs, epsilon)
    ar = np.arange(obs.shape[0])
    return actions, q, kc_cache, q[ar, actions]


def test_act_shapes_and_range():
    brain = make_brain()
    obs = np.zeros((5, 5), dtype=np.float32)
    actions, q, kc_cache = brain.act(obs, epsilon=0.0)
    assert actions.shape == (5,)
    assert set(np.unique(actions)) <= {0, 1}
    assert q.shape == (5, 2)
    assert len(kc_cache) == 2
    assert kc_cache[0].shape == (5, 64)
    assert kc_cache[0].sum(axis=1).max() <= 8


def test_kc_sparsity():
    brain = make_brain()
    obs = np.random.default_rng(0).uniform(-1, 1, (10, 5)).astype(np.float32)
    _, _, kc_cache = brain.act(obs, epsilon=0.0)
    for kc in kc_cache:
        assert np.all(kc.sum(axis=1) == brain.cfg.kc_active)


def test_weights_bounded_after_learning():
    brain = make_brain()
    obs = np.random.default_rng(1).uniform(-1, 1, (16, 5)).astype(np.float32)
    for _ in range(50):
        actions, _, kc_cache, q_taken = rollout(brain, obs, 0.5)
        rewards = np.full(16, 0.5, dtype=np.float32)
        q_next = brain.values(obs).max(axis=1)
        brain.learn(actions, kc_cache, rewards, q_taken, q_next,
                    np.zeros(16, dtype=bool))
        assert brain.w_approach.min() >= 0.0
        assert brain.w_approach.max() <= brain.cfg.w0
        assert brain.w_avoid.min() >= 0.0
        assert brain.w_avoid.max() <= brain.cfg.w0


def test_positive_rpe_unlearns_avoidance():
    brain = make_brain()
    obs = np.zeros((4, 5), dtype=np.float32)
    actions, _, kc_cache, q_taken = rollout(brain, obs, 0.0)
    before = brain.w_avoid.copy()
    rewards = np.full(4, 5.0, dtype=np.float32)  # strongly positive RPE
    q_next = np.zeros(4, dtype=np.float32)
    brain.learn(actions, kc_cache, rewards, q_taken, q_next,
                np.zeros(4, dtype=bool))
    # active synapses onto avoid MBONs must not increase
    assert (brain.w_avoid <= before + 1e-6).all()


def test_terminal_death_has_negative_rpe():
    brain = make_brain()
    obs = np.zeros((4, 5), dtype=np.float32)
    actions, _, kc_cache, q_taken = rollout(brain, obs, 0.0)
    rewards = np.full(4, -1.0, dtype=np.float32)
    q_next = np.full(4, 3.0, dtype=np.float32)  # would be ignored at terminal
    rpe = brain.learn(actions, kc_cache, rewards, q_taken, q_next,
                      np.ones(4, dtype=bool))
    assert (rpe < 0).all()


def test_alive_mask_blocks_updates():
    brain = make_brain()
    obs = np.zeros((4, 5), dtype=np.float32)
    actions, _, kc_cache, q_taken = rollout(brain, obs, 0.0)
    before = (brain.w_approach.copy(), brain.w_avoid.copy())
    rewards = np.full(4, -1.0, dtype=np.float32)
    q_next = np.zeros(4, dtype=np.float32)
    brain.learn(actions, kc_cache, rewards, q_taken, q_next,
                np.ones(4, dtype=bool), alive=np.zeros(4, dtype=bool))
    assert np.array_equal(brain.w_approach, before[0])
    assert np.array_equal(brain.w_avoid, before[1])


def test_save_load_roundtrip(tmp_path):
    brain = make_brain()
    path = str(tmp_path / "w.npz")
    brain.save(path)
    other = make_brain()
    other.load(path)
    assert np.array_equal(brain.w_approach, other.w_approach)
    assert np.array_equal(brain.w_avoid, other.w_avoid)
