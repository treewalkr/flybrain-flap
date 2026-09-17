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
    b.feat_mean = np.zeros(b.n_sensory, np.float32)
    b.feat_scale = np.ones(b.n_sensory, np.float32)
    kc = b.kc_code(b.encode(feats), np.zeros((64, 2), np.float32))
    assert kc.shape == (64, 200)
    assert (kc.sum(axis=1) == 40).all()  # exactly top-k per bird


def test_bandit_learns_the_good_action():
    """Two-armed bandit: action 1 gives +1, action 0 gives -1. After training,
    the greedy choice must be action 1."""
    b = RealMB(tiny_circuit(seed=1), RealMBConfig(kc_active=40, seed=5, alpha=0.05))
    rng = np.random.default_rng(7)
    feats = rng.uniform(-1, 1, (512, 5)).astype(np.float32)
    b.calibrate(b.encode(feats))
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
    assert (q[:, 1] > q[:, 0]).mean() > 0.8, f"q={q.mean(0)}"


def test_eligibility_trace_gives_delayed_credit():
    """A reward arriving 10 ticks after the choice must reach the chosen
    action's synapses with traces (lambda=0.9), and barely without (lambda=0).
    Codes are constructed directly (disjoint, connected rows) so this tests
    the trace mechanism itself, independent of the encoding."""
    results = {}
    for lam in (0.0, 0.9):
        b = RealMB(tiny_circuit(seed=3), RealMBConfig(kc_active=40, seed=13, alpha=0.1,
                                                      trace_lambda=lam))
        rng = np.random.default_rng(4)
        feats = rng.uniform(-1, 1, (256, 5)).astype(np.float32)
        b.calibrate(b.encode(feats))

        # disjoint synthetic codes over KCs that actually have MBON synapses
        rows = rng.permutation(np.flatnonzero(b.w0.sum(1) > 0))
        n_code = 20
        code1_row, code0_row = rows[:n_code], rows[n_code:2 * n_code]
        kc1 = np.zeros((16, b.n_kc), bool); kc1[:, code1_row] = True
        kc0 = np.zeros((16, b.n_kc), bool); kc0[:, code0_row] = True
        cache = [kc0, kc1]

        b.learn(np.ones(16, np.int64), cache, np.zeros(16, np.float32),
                np.zeros(16, np.float32), np.zeros(16, np.float32),
                np.zeros(16, bool))
        w_after = b.w.copy()
        for _ in range(9):  # ticks 1..9: action 0, no reward
            b.learn(np.zeros(16, np.int64), cache, np.zeros(16, np.float32),
                    np.zeros(16, np.float32), np.zeros(16, np.float32),
                    np.zeros(16, bool))
        # tick 10: positive RPE — must reach the tick-0 (action 1) synapses
        b.learn(np.zeros(16, np.int64), cache, np.ones(16, np.float32),
                np.full(16, 0.5), np.zeros(16, np.float32), np.zeros(16, bool))
        moved = np.abs(b.w - w_after)[code1_row]
        results[lam] = float(moved.mean())
    assert results[0.9] > results[0.0] * 3 + 1e-4, results


def test_kc_input_fast_path_exact():
    """The candidate-only fast path in _kc_input must be bit-identical to a
    plain min-reduceat over all edges (drives, codes and all)."""
    b = RealMB(tiny_circuit(seed=5), RealMBConfig(kc_active=30, seed=7))
    rng = np.random.default_rng(11)
    feats = rng.uniform(-1, 1, (128, 5)).astype(np.float32)
    b.calibrate(b.encode(feats))

    def slow(pn_act):
        out = np.empty((pn_act.shape[0], b.n_kc), np.float32)
        edge_vals = pn_act[:, b.edge_pre]
        out[:, b.kc_group_kc] = np.minimum.reduceat(edge_vals, b.kc_group_start, axis=1)
        out[:, ~b.kc_has_input] = -1e9
        return out + b.kc_bias

    enc = b.encode(feats)
    fa = np.repeat(enc[:, None, :], 2, 1).reshape(-1, b.n_sensory)
    ca = np.repeat(b.action_codes[None], 128, 0).reshape(-1, 2)
    pn = b._pn(fa, ca)
    np.testing.assert_array_equal(b._kc_input(pn), slow(pn))


def test_save_load_roundtrip(tmp_path):
    b = RealMB(tiny_circuit(seed=2), RealMBConfig(kc_active=40, seed=9))
    feats = np.random.default_rng(1).uniform(-1, 1, (64, 5)).astype(np.float32)
    b.calibrate(b.encode(feats))
    b.w = b.w * 0.5  # pretend some learning happened
    b.save(tmp_path / "w.npz")
    b2 = RealMB(tiny_circuit(seed=2), RealMBConfig(kc_active=40, seed=9))
    b2.load(tmp_path / "w.npz")
    obs = rng = np.random.default_rng(2).uniform(-1, 1, (8, 5)).astype(np.float32)
    np.testing.assert_allclose(b.values(obs), b2.values(obs), rtol=1e-5)


def test_weights_stay_bounded():
    b = RealMB(tiny_circuit(seed=4), RealMBConfig(kc_active=40, seed=11, alpha=0.5))
    feats = np.random.default_rng(3).uniform(-1, 1, (64, 5)).astype(np.float32)
    b.calibrate(b.encode(feats))
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
