"""Tests for the real-connectome mushroom body (synthetic small circuit)."""

from __future__ import annotations

import numpy as np

from flybrain_flap.realbrain import RealMB, RealMBConfig
from flybrain_flap.connectome import APPROACH, AVOID


def tiny_circuit(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_pn, n_kc, n_mbon = 24, 200, 6
    pn_kc = (rng.random((n_pn, n_kc)) < 6 / n_pn).astype(np.float32) * \
        rng.integers(1, 4, (n_pn, n_kc)).astype(np.float32)
    kc_mbon = (rng.random((n_kc, n_mbon)) < 12 / n_kc).astype(np.float32)
    group = np.full(n_mbon, AVOID, dtype=np.int64)
    group[::2] = APPROACH
    return {
        "pn_body": np.arange(n_pn), "kc_body": np.arange(n_kc),
        "mbon_body": np.arange(n_mbon), "mbon_group": group,
        "pn_kc": pn_kc, "kc_mbon": kc_mbon,
        "pn_pos": rng.uniform(0, 100, (n_pn, 3)).astype(np.float32),
        "kc_pos": rng.uniform(0, 100, (n_kc, 3)).astype(np.float32),
        "mbon_pos": rng.uniform(0, 100, (n_mbon, 3)).astype(np.float32),
    }


def test_code_is_sparse_and_topk():
    b = RealMB(tiny_circuit(), RealMBConfig(kc_active=40, seed=3))
    feats = np.random.default_rng(0).uniform(-1, 1, (64, 5)).astype(np.float32)
    b.feat_mean, b.feat_scale = feats.mean(0), feats.std(0)
    kc = b.kc_code(feats, np.zeros((64, 2), np.float32))
    assert kc.shape == (64, 200)
    assert (kc.sum(axis=1) == 40).all()  # exactly top-k per bird


def test_bandit_learns_the_good_action():
    """Two-armed bandit: action 1 gives +1, action 0 gives -1. After training,
    the greedy choice must be action 1."""
    b = RealMB(tiny_circuit(seed=1), RealMBConfig(kc_active=40, seed=5, alpha=0.05))
    rng = np.random.default_rng(7)
    feats = rng.uniform(-1, 1, (512, 5)).astype(np.float32)
    b.calibrate(feats)
    assert b.readout[b.approach & (b.w0.sum(0) > 0)].sum() > 0
    assert b.readout[b.avoid & (b.w0.sum(0) > 0)].sum() < 0

    obs = rng.uniform(-1, 1, (32, 5)).astype(np.float32)
    for _ in range(300):
        actions, q, kc = b.act(obs, epsilon=0.2)
        rewards = np.where(actions == 1, 1.0, -1.0).astype(np.float32)
        q_next = b.values(obs).max(axis=1)  # bandit: same state value
        b.learn(actions, kc, rewards, q[np.arange(32), actions], q_next,
                dones=np.zeros(32, bool))
    actions, q, _ = b.act(obs, epsilon=0.0)
    assert (q[:, 1] > q[:, 0]).mean() > 0.9, f"q={q.mean(0)}"


def test_save_load_roundtrip(tmp_path):
    b = RealMB(tiny_circuit(seed=2), RealMBConfig(kc_active=40, seed=9))
    feats = np.random.default_rng(1).uniform(-1, 1, (64, 5)).astype(np.float32)
    b.calibrate(feats)
    b.w = b.w * 0.5  # pretend some learning happened
    b.save(tmp_path / "w.npz")
    b2 = RealMB(tiny_circuit(seed=2), RealMBConfig(kc_active=40, seed=9))
    b2.load(tmp_path / "w.npz")
    obs = rng = np.random.default_rng(2).uniform(-1, 1, (8, 5)).astype(np.float32)
    np.testing.assert_allclose(b.values(obs), b2.values(obs), rtol=1e-5)


def test_weights_stay_bounded():
    b = RealMB(tiny_circuit(seed=4), RealMBConfig(kc_active=40, seed=11, alpha=0.5))
    feats = np.random.default_rng(3).uniform(-1, 1, (64, 5)).astype(np.float32)
    b.calibrate(feats)
    obs = feats[:32]
    for step in range(200):
        r = 1.0 if step % 2 == 0 else -1.0
        actions, q, kc = b.act(obs, epsilon=0.5)
        rewards = np.full(32, r, dtype=np.float32)
        b.learn(actions, kc, rewards, q[np.arange(32), actions],
                b.values(obs).max(1), np.zeros(32, bool))
    floor, ceiling = b.saturation()
    assert floor + ceiling <= 1.0
    assert (b.w >= 0).all() and (b.w <= b.w0 + 1e-6).all()
