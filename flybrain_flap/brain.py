"""Mushroom body: PN -> KC (coincidence code) -> MBON, dopamine RPE rule.

See docs/BRAIN.md for the circuit and the plasticity rule.

The front end mirrors the fly's antennal lobe -> calyx expansion:
projection neurons are feature detectors with broad tuning curves, and each
Kenyon cell is a coincidence detector over a few random PNs — it fires only
when all of its PNs are active. This gives the sparse, high-dimensional
code the real mushroom body uses (~5% of KCs active).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MBConfig:
    num_features: int
    num_actions: int = 2
    num_bins: int = 7          # PN tuning curves per feature
    bin_width: float = 0.6     # tuning width in feature units (features ~[-1,1])
    num_kc: int = 1024         # total, split into per-action pools
    kc_active: int = 16        # active KCs per pool
    pn_per_kc: int = 3         # PNs a KC coincides on
    num_mbon_per_class: int = 8
    w0: float = 1.0            # max KC->MBON weight
    eta: float = 0.02          # dopamine depression rate
    rec_eta: float | None = None  # recovery under below-baseline DA (None = eta)
    gamma: float = 0.95        # discount
    seed: int | None = None


class MushroomBody:
    """Batched mushroom body over N parallel agents.

    Kenyon cells are split into one pool per candidate action, so the value
    of each action is read from its own synapses. This stands in for the
    real circuit's action-specific compartments.
    """

    def __init__(self, cfg: MBConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        m = cfg.num_mbon_per_class
        assert cfg.num_kc % cfg.num_actions == 0
        self.pool = cfg.num_kc // cfg.num_actions

        # PN tuning curve centers, one row of bins per feature
        self.pn_centers = np.linspace(-1.0, 1.0, cfg.num_bins)
        self.num_pn = cfg.num_features * cfg.num_bins

        # each KC coincides on pn_per_kc PNs from distinct features
        kc_feats = np.stack([
            rng.choice(cfg.num_features, size=cfg.pn_per_kc, replace=False)
            for _ in range(self.pool)
        ])
        kc_bins = rng.integers(0, cfg.num_bins, size=(self.pool, cfg.pn_per_kc))
        # (feature, bin) index per KC input, for gather
        self.kc_feat = kc_feats.T          # (pn_per_kc, pool)
        self.kc_bin = kc_bins.T

        # learned KC -> MBON weights, [0, w0], starting fully potentiated
        # (dopamine depresses; below-baseline dopamine restores)
        self.w_approach = rng.uniform(
            0.9 * cfg.w0, cfg.w0, (cfg.num_kc, m)
        ).astype(np.float32)
        self.w_avoid = rng.uniform(
            0.9 * cfg.w0, cfg.w0, (cfg.num_kc, m)
        ).astype(np.float32)

    # -- forward ---------------------------------------------------------

    def _pn_activity(self, obs: np.ndarray) -> np.ndarray:
        """obs: (N, F) ~[-1,1] -> PN tuning responses (N, F, num_bins)."""
        # triangular tuning curves: two adjacent bins respond per feature
        d = np.abs(obs[:, :, None] - self.pn_centers[None, None, :])
        return np.clip(1.0 - d / self.cfg.bin_width, 0.0, 1.0)

    def _kc_activity(self, obs: np.ndarray, action: int) -> np.ndarray:
        """obs: (N, F) -> bool (N, num_kc); top-k coincidence per pool."""
        pn = self._pn_activity(obs)  # (N, F, B)
        n = obs.shape[0]
        # soft-AND: the weakest PN input gates the KC
        responses = pn[
            np.arange(n)[:, None, None],
            self.kc_feat[None],          # (pn_per_kc, pool)
            self.kc_bin[None],
        ].min(axis=1)                    # (N, pool)
        k = self.cfg.kc_active
        idx = np.argpartition(responses, -k, axis=1)[:, -k:]
        pool_kc = np.zeros(responses.shape, dtype=bool)
        np.put_along_axis(pool_kc, idx, True, axis=1)
        kc = np.zeros((n, self.cfg.num_kc), dtype=bool)
        kc[:, action * self.pool : (action + 1) * self.pool] = pool_kc
        return kc

    def _q_from_kc(self, kc: np.ndarray) -> np.ndarray:
        """Net valence of a KC code, normalized to roughly [-1, 1]."""
        app = (kc.astype(np.float32) @ self.w_approach).sum(axis=1)
        avd = (kc.astype(np.float32) @ self.w_avoid).sum(axis=1)
        return (app - avd) / (self.cfg.kc_active * self.cfg.num_mbon_per_class)

    def values(self, obs: np.ndarray) -> np.ndarray:
        """Values for all actions. obs: (N, F). Returns (N, num_actions)."""
        q = np.zeros((obs.shape[0], self.cfg.num_actions), dtype=np.float32)
        for a in range(self.cfg.num_actions):
            q[:, a] = self._q_from_kc(self._kc_activity(obs, a))
        return q

    def act(
        self,
        obs: np.ndarray,
        epsilon: float,
        explore_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Returns actions, per-action values (N, A), KC activity per action.

        kc_cache[a] is the KC boolean activity for action a, reused by learn().
        explore_probs biases the action sampled while exploring (None = uniform).
        """
        n = obs.shape[0]
        na = self.cfg.num_actions
        kc_cache = []
        q = np.zeros((n, na), dtype=np.float32)
        for a in range(na):
            kc = self._kc_activity(obs, a)
            kc_cache.append(kc)
            q[:, a] = self._q_from_kc(kc)

        greedy = q.argmax(axis=1)
        explore = self.rng.random(n) < epsilon
        if explore_probs is None:
            random_actions = self.rng.integers(0, na, n)
        else:
            random_actions = self.rng.choice(na, size=n, p=explore_probs)
        actions = np.where(explore, random_actions, greedy)
        return actions.astype(np.int64), q, kc_cache

    # -- learning --------------------------------------------------------

    def learn(
        self,
        actions: np.ndarray,
        kc_cache: list[np.ndarray],
        rewards: np.ndarray,
        q_taken: np.ndarray,
        q_next: np.ndarray,
        dones: np.ndarray,
        alive: np.ndarray | None = None,
    ) -> np.ndarray:
        """One dopamine update from cached forward passes.

        kc_cache[a][i] is the KC code agent i used when it chose action a;
        q_taken / q_next are the values from that pass and the next pass.
        "alive" masks agents to learn from (None = all).
        """
        cfg = self.cfg
        rec_eta = cfg.rec_eta if cfg.rec_eta is not None else cfg.eta
        target = rewards + cfg.gamma * q_next * (~dones)
        rpe = target - q_taken

        learn = np.ones_like(rpe, dtype=bool) if alive is None else alive
        pos = np.clip(rpe, 0.0, None) * learn   # above-baseline dopamine
        neg = np.clip(-rpe, 0.0, None) * learn  # below-baseline dopamine

        for a in range(self.cfg.num_actions):
            mask = (actions == a) & learn
            if not mask.any():
                continue
            kc_f = kc_cache[a][mask].astype(np.float32)  # (n_mask, num_kc)
            scale = 1.0 / max(1, int(mask.sum()))  # mean over the batch

            # PAM carries +RPE: depresses avoid MBON synapses; dopamine
            # below baseline lets them recover (Handler 19, Bennett 21)
            self.w_avoid -= (kc_f.T @ (pos[mask] * cfg.eta * scale))[:, None]
            self.w_avoid += (kc_f.T @ (neg[mask] * rec_eta * scale))[:, None]
            # PPL1 carries -RPE onto approach MBON synapses, same logic
            self.w_approach -= (kc_f.T @ (neg[mask] * cfg.eta * scale))[:, None]
            self.w_approach += (kc_f.T @ (pos[mask] * rec_eta * scale))[:, None]

            np.clip(self.w_approach, 0.0, cfg.w0, out=self.w_approach)
            np.clip(self.w_avoid, 0.0, cfg.w0, out=self.w_avoid)

        return rpe

    # -- persistence -----------------------------------------------------

    def save(self, path: str):
        np.savez(
            path,
            kc_feat=self.kc_feat,
            kc_bin=self.kc_bin,
            pn_centers=self.pn_centers,
            w_approach=self.w_approach,
            w_avoid=self.w_avoid,
        )

    def load(self, path: str):
        data = np.load(path)
        self.kc_feat = data["kc_feat"]
        self.kc_bin = data["kc_bin"]
        self.pn_centers = data["pn_centers"]
        self.w_approach = data["w_approach"]
        self.w_avoid = data["w_avoid"]
