"""Mushroom body wired from the real MaleCNS connectome.

Same role as `brain.MushroomBody` (same act/learn/values/observe/save API, a
drop-in for `train.py` and `game.py`) but the fixed wiring is the fly's:

  - 686 real ALPNs (random sparse projections of the game features, plus
    action-code inputs so a single mixed KC code can express per-action values)
  - the real PN -> KC synapse-count matrix (the fixed expansion; coincidence
    code = weighted mean of a KC's PN inputs, top-k winner-take-all)
  - 97 real MBONs; the only plastic synapses are the real KC -> MBON counts,
    bounded to [0, w0]
  - dopamine gated per MBON group: PAM compartments (avoid MBONs) carry
    +RPE, PPL1 compartments (approach MBONs) carry -RPE

Nothing here is learned except the KC -> MBON weights. See
`connectome.py` for how the circuit is extracted, docs/BRAIN.md for the rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .connectome import APPROACH, AVOID, load as load_circuit


@dataclass
class RealMBConfig:
    kc_active: int = 200       # winner-take-all code size (of ~3.8k KCs with input)
    n_action_dims: int = 2     # action-code dimensions fed to a fraction of PNs
    pn_action_fraction: float = 0.5  # fraction of PNs that also read an action code
    pn_quantile: float = 0.5   # PN activation threshold at this quantile of input
    target_action_overlap: float = 0.5  # KC-code overlap between actions
    num_bins: int = 7          # triangular tuning bins per feature (antennal lobe)
    bin_width: float = 0.6     # tuning width in feature units (features ~[-1,1])
    pn_state_inputs: int = 8   # tuning bins each real PN reads (sparse, like ALPNs)
    gain: float = 1.0          # value scale (approach-minus-avoid drive)
    alpha: float = 0.02        # target effective dV per unit RPE
    gamma: float = 0.99        # discount
    binary_readout: bool = True  # KC->MBON base weights: connectivity (True)
                                 # or raw synapse counts (False; count-derived
                                 # ceilings span 1000x and hub MBONs dominate)
    seed: int | None = None


class RealMB:
    """Batched mushroom body over N parallel agents, real connectome wiring."""

    def __init__(self, circuit: dict, cfg: RealMBConfig | None = None):
        self.cfg = cfg or RealMBConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        c = circuit
        self.circuit = c

        pn_kc = c["pn_kc"].astype(np.float32)
        self.n_pn, self.n_kc = pn_kc.shape
        self.n_mbon = c["kc_mbon"].shape[1]
        self.n_features = 5  # flappy observation width

        # fixed expansion: weighted mean of a KC's PN inputs
        col = pn_kc.sum(axis=0, keepdims=True)
        self.kc_has_input = col.ravel() > 0
        self.pn_kc = pn_kc / np.maximum(col, 1.0)
        self.kc_bias = np.where(self.kc_has_input, 0.0, -1e9).astype(np.float32)

        # real MBON grouping
        group = c["mbon_group"].astype(np.int64)
        self.mbon_group = group
        self.approach = group == APPROACH
        self.avoid = group == AVOID

        # antennal lobe: features enter as coarse triangular tuning over bins
        # (like the abstract brain's PNs), then every real PN reads a sparse
        # random subset of the bins — the fly's glomerular wiring
        self.num_bins = cfg.num_bins
        self.bin_width = cfg.bin_width
        self.n_sensory = 5 * cfg.num_bins
        centers = np.linspace(-1.0, 1.0, cfg.num_bins, dtype=np.float32)
        self.bin_centers = np.tile(centers, 5)

        # plastic weights start at the real connectivity, optionally scaled by
        # synapse counts (normalized in calibrate)
        kc_mbon = c["kc_mbon"].astype(np.float32)
        if cfg.binary_readout:
            kc_mbon = (kc_mbon > 0).astype(np.float32)
        self.w0_base = kc_mbon

        # random sensory projection: every real PN reads pn_state_inputs random
        # tuning bins; a fraction also reads one action-code dimension
        cfg = self.cfg
        u = np.zeros((self.n_pn, self.n_sensory), dtype=np.float32)
        for i in range(self.n_pn):
            cols = self.rng.choice(self.n_sensory, cfg.pn_state_inputs, replace=False)
            u[i, cols] = self.rng.standard_normal(cfg.pn_state_inputs)
        self.pn_u = u
        reads = self.rng.random(self.n_pn) < cfg.pn_action_fraction
        act_col = self.rng.integers(0, cfg.n_action_dims, self.n_pn)
        act_sign = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), self.n_pn)
        self.pn_action_mask = np.zeros((self.n_pn, cfg.n_action_dims), dtype=np.float32)
        self.pn_action_mask[np.arange(self.n_pn)[reads], act_col[reads]] = act_sign[reads]
        self.action_codes = np.eye(cfg.n_action_dims, dtype=np.float32)[:2] \
            if cfg.n_action_dims >= 2 else None
        assert self.action_codes is not None, "need at least 2 action dims"

        # calibration results
        self.feat_mean = np.zeros(self.n_sensory, dtype=np.float32)
        self.feat_scale = np.ones(self.n_sensory, dtype=np.float32)
        self.pn_theta = np.zeros(self.n_pn, dtype=np.float32)
        self.pn_scale = np.ones(self.n_pn, dtype=np.float32)
        self.action_gain = 1.0
        self.w0 = self.w0_base.copy()
        self.w = self.w0.copy()
        self.readout = np.zeros(self.n_mbon, dtype=np.float32)
        self.eta = 0.0
        self.calibrated = False

    # ---------------------------------------------------------------- sensory
    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Raw observations (N, 5) -> antennal-lobe code (N, 5*num_bins):
        triangular tuning per feature, like the abstract brain's PNs."""
        x = obs[:, None, :5]  # (N, 1, F)
        d = np.abs(x - self.bin_centers.reshape(1, -1, 5))
        r = np.maximum(1.0 - d / self.bin_width, 0.0)
        return r.max(axis=2).reshape(obs.shape[0], self.n_sensory).astype(np.float32)

    def _pn_pre(self, features: np.ndarray, codes: np.ndarray) -> np.ndarray:
        x = (features - self.feat_mean) / self.feat_scale
        return x @ self.pn_u.T + self.action_gain * (codes @ self.pn_action_mask.T)

    def _pn(self, features: np.ndarray, codes: np.ndarray) -> np.ndarray:
        return np.maximum(self._pn_pre(features, codes) - self.pn_theta, 0.0) * self.pn_scale

    def kc_code(self, features: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """(B, F) features + (B, A_dims) action codes -> (B, n_kc) bool top-k code."""
        h = self._pn(features, codes) @ self.pn_kc + self.kc_bias
        k = min(self.cfg.kc_active, h.shape[1])
        idx = np.argpartition(h, -k, axis=1)[:, -k:]
        kc = np.zeros(h.shape, dtype=bool)
        np.put_along_axis(kc, idx, True, axis=1)
        return kc

    def kc_all_actions(self, obs: np.ndarray) -> np.ndarray:
        """(N, 5) raw obs -> (N, A, n_kc) bool codes for every candidate action."""
        features = self.encode(obs)
        n = features.shape[0]
        f = np.repeat(features[:, None, :], 2, axis=1).reshape(-1, self.n_sensory)
        c = np.repeat(self.action_codes[None, :, :], n, axis=0).reshape(-1, 2)
        return self.kc_code(f, c).reshape(n, 2, self.n_kc)

    # ------------------------------------------------------------------ value
    def drives(self, kc: np.ndarray) -> np.ndarray:
        return (kc.astype(np.float32) @ self.w) / self.cfg.kc_active

    def _q_from_kc(self, kc: np.ndarray) -> np.ndarray:
        return self.drives(kc) @ self.readout

    def values(self, obs: np.ndarray) -> np.ndarray:
        """Values for all actions. obs: (N, F). Returns (N, 2)."""
        return self._q_from_kc(self.kc_all_actions(obs))

    def act(
        self,
        obs: np.ndarray,
        epsilon: float,
        explore_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Same contract as MushroomBody.act: actions (N,), q (N, 2),
        kc_cache[a] = (N, n_kc) bool code for action a."""
        n = obs.shape[0]
        kcs = self.kc_all_actions(obs)
        kc_cache = [kcs[:, 0], kcs[:, 1]]
        q = np.stack([self._q_from_kc(kc_cache[0]), self._q_from_kc(kc_cache[1])], axis=1)

        greedy = q.argmax(axis=1)
        explore = self.rng.random(n) < epsilon
        if explore_probs is None:
            random_actions = self.rng.integers(0, 2, n)
        else:
            random_actions = self.rng.choice(2, size=n, p=explore_probs)
        actions = np.where(explore, random_actions, greedy)
        return actions.astype(np.int64), q, kc_cache

    # -------------------------------------------------------------- calibrate
    def calibrate(self, features: np.ndarray) -> dict:
        """Fix normalization, PN thresholds/gains, action gain and the KC->MBON
        normalization from a batch of states. Nothing is learned from reward."""
        cfg = self.cfg
        m = features.shape[0]
        self.feat_mean = features.mean(axis=0)
        self.feat_scale = np.maximum(features.std(axis=0), 0.05)
        codes = self.action_codes[self.rng.integers(0, 2, m)]

        def set_pn(gain: float) -> None:
            self.action_gain = gain
            self.pn_theta = np.zeros(self.n_pn, dtype=np.float32)
            self.pn_scale = np.ones(self.n_pn, dtype=np.float32)
            z = self._pn_pre(features, codes)
            self.pn_theta = np.quantile(z, cfg.pn_quantile, axis=0).astype(np.float32)
            p = np.maximum(z - self.pn_theta, 0.0)
            self.pn_scale = (1.0 / np.maximum(p.mean(axis=0), 1e-6)).astype(np.float32)

        def overlap() -> float:
            sub = features[: min(m, 1024)]
            kc = self.kc_all_actions(sub)
            inter = (kc[:, 0] & kc[:, 1]).sum(axis=1).astype(np.float32)
            return float((inter / cfg.kc_active).mean())

        lo, hi = 0.0, 64.0
        for _ in range(17):
            mid = 0.5 * (lo + hi)
            set_pn(mid)
            if overlap() > cfg.target_action_overlap:
                lo = mid
            else:
                hi = mid
        set_pn(0.5 * (lo + hi))

        # KC->MBON normalization: each connected MBON's mean drive over
        # calibration codes is 1; readout balances the two groups (start at 0)
        kc = self.kc_code(features, codes)
        drive0 = self.drives(kc).mean(axis=0)
        connected = drive0 > 0
        self.w0 = self.w0_base * np.where(connected, 1.0 / np.maximum(drive0, 1e-9), 0.0)
        self.w = self.w0.copy()
        n_app = int((self.approach & connected).sum())
        n_avd = int((self.avoid & connected).sum())
        self.readout = np.zeros(self.n_mbon, dtype=np.float32)
        self.readout[self.approach & connected] = cfg.gain / max(n_app, 1)
        self.readout[self.avoid & connected] = -cfg.gain / max(n_avd, 1)

        # effective TD step: eta from alpha = dV per unit RPE for a fixed code
        conn = (self.w0 > 0).astype(np.float32)
        f = (kc.astype(np.float32) @ conn).mean(axis=0) / cfg.kc_active
        eff_app = float((f * np.maximum(self.readout, 0)).sum())
        eff_avd = float((f * np.maximum(-self.readout, 0)).sum())
        self.eta = cfg.alpha / max(eff_app + eff_avd, 1e-9)
        self.calibrated = True
        return {"action_gain": self.action_gain, "action_overlap": overlap(),
                "eta": self.eta, "mbons_connected": int(connected.sum()),
                "approach_mbons_used": n_app, "avoid_mbons_used": n_avd}

    # --------------------------------------------------------------- learning
    def dopamine_update(self, kc: np.ndarray, rpe: np.ndarray) -> float:
        """dw = -eta * KC * (DAN_c - baseline_c) summed over a batch.

        PAM (reward) compartments act on avoid MBONs with +RPE; PPL1
        (punishment) compartments act on approach MBONs with -RPE. Dopamine
        above baseline with an active KC depresses the synapse; below baseline
        lets it recover (Handler et al. 2019; Bennett et al. 2021). Weights
        stay in [0, w0]."""
        coincidence = kc.astype(np.float32).T @ rpe.astype(np.float32)  # (n_kc,)
        signed = np.where(self.avoid[None, :], coincidence[:, None],
                          np.where(self.approach[None, :], -coincidence[:, None], 0.0))
        before = self.w
        self.w = np.minimum(np.maximum(self.w - self.eta * signed, 0.0), self.w0)
        exist = self.w0 > 0
        return float(np.abs(self.w - before)[exist].mean()) if exist.any() else 0.0

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
        """One dopamine update from cached forward passes (same contract as
        MushroomBody.learn). Returns the per-agent RPE."""
        target = rewards + self.cfg.gamma * q_next * (~dones)
        rpe = target - q_taken
        learn = np.ones_like(rpe, dtype=bool) if alive is None else alive
        for a in (0, 1):
            mask = (actions == a) & learn
            if mask.any():
                # zero out non-learning agents so they contribute nothing
                self.dopamine_update(kc_cache[a][mask], rpe[mask])
        return rpe

    # ------------------------------------------------------------ inspect/save
    def saturation(self) -> tuple[float, float]:
        exist = self.w0 > 0
        return (float((self.w[exist] <= 0).mean()), float((self.w[exist] >= self.w0[exist]).mean()))

    def observe(self, obs: np.ndarray) -> dict:
        """Activity snapshot for visualization (batch obs (N, F))."""
        kcs = self.kc_all_actions(obs)
        d = self.drives(kcs)  # (N, A, n_mbon)
        q = np.einsum("nam,m->na", d, self.readout)
        return {
            "pn": self._pn(obs, np.zeros((obs.shape[0], 2), dtype=np.float32)),
            "kc": [kcs[:, 0], kcs[:, 1]],
            "app": np.stack([d[:, a][..., self.approach] for a in (0, 1)], axis=1),
            "avd": np.stack([d[:, a][..., self.avoid] for a in (0, 1)], axis=1),
            "q": q,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w=self.w, w0=self.w0, readout=self.readout, mbon_group=self.mbon_group,
            pn_u=self.pn_u, pn_action_mask=self.pn_action_mask,
            pn_theta=self.pn_theta, pn_scale=self.pn_scale,
            feat_mean=self.feat_mean, feat_scale=self.feat_scale,
            action_codes=self.action_codes,
            action_gain=np.array(self.action_gain), eta=np.array(self.eta),
            gain=np.array(self.cfg.gain), kc_active=np.array(self.cfg.kc_active),
            alpha=np.array(self.cfg.alpha), gamma=np.array(self.cfg.gamma),
            seed=np.array(self.cfg.seed if self.cfg.seed is not None else -1),
        )

    def load(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as d:
            self.w = d["w"]; self.w0 = d["w0"]; self.readout = d["readout"]
            self.mbon_group = d["mbon_group"]
            self.approach = self.mbon_group == APPROACH
            self.avoid = self.mbon_group == AVOID
            self.pn_u = d["pn_u"]; self.pn_action_mask = d["pn_action_mask"]
            self.pn_theta = d["pn_theta"]; self.pn_scale = d["pn_scale"]
            self.feat_mean = d["feat_mean"]; self.feat_scale = d["feat_scale"]
            self.action_codes = d["action_codes"]
            self.action_gain = float(d["action_gain"]); self.eta = float(d["eta"])
            self.cfg.gain = float(d["gain"]); self.cfg.kc_active = int(d["kc_active"])
            self.cfg.alpha = float(d["alpha"]); self.cfg.gamma = float(d["gamma"])
            seed = int(d["seed"])
            self.cfg.seed = None if seed < 0 else seed
        self.calibrated = True


def calibrate_from_env(brain: RealMB, env, steps: int = 4096) -> dict:
    """Collect a calibration batch of live-bird states with a random policy."""
    obs = env.reset()
    batch = np.zeros((steps, brain.n_sensory), dtype=np.float32)
    got = 0
    while got < steps:
        live = obs[env.alive]
        take = min(len(live), steps - got)
        batch[got:got + take] = brain.encode(live[:take])
        got += take
        obs, _, _ = env.step(env.rng.integers(0, 2, obs.shape[0]))
        if not env.alive.any():
            obs = env.reset()
    return brain.calibrate(batch)
