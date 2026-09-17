"""3D brain-activity renderer: real neuron skeletons, coloured by live activity.

    .venv/bin/python -m flybrain_flap.render3d --weights runs/real04/weights.npz \
        --out brain.mp4 [--ticks 600] [--fps 30] [--orbit 10]

Renders the MaleCNS skeletons of the mushroom-body populations (PN, KC, MBON,
DAN) as polylines inside the JRCFIB2022M brain shell. Every frame, the brain
plays the game greedily; each neuron is coloured by its activity:

  PN   tuning response
  KC   membership in the top-k code (binary)
  MBON  approach/avoid drive magnitude
  DAN  signed reward-prediction error (PAM: +RPE, PPL1: -RPE)

Requires `python -m flybrain_flap.fetch_connectome` data and a trained RealMB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .connectome import APPROACH, AVOID, load as load_circuit
from .env import FlappyEnv
from .fetch_connectome import NPZ_DIR, PLY_DIR

BRAIN_CENTER = np.array([386.0, 228.0, 211.0], dtype=np.float32)  # um, shell bounds

# activity -> colour ramp: dim grey-blue -> warm orange -> hot white-yellow
STOPS = np.array([
    [0.18, 0.20, 0.28],
    [0.45, 0.45, 0.45],
    [0.95, 0.55, 0.15],
    [1.00, 0.90, 0.55],
    [1.00, 1.00, 0.95],
], dtype=np.float64)


def activity_color(a: np.ndarray) -> np.ndarray:
    """(N,) activity in [0, 1] -> (N, 3) colours via the ramp."""
    a = np.clip(a, 0.0, 1.0)
    x = a * (len(STOPS) - 1)
    i = np.clip(x.astype(int), 0, len(STOPS) - 2)
    f = (x - i)[:, None]
    return STOPS[i] * (1 - f) + STOPS[i + 1] * f


class SkeletonCloud:
    """All skeletons of the four populations merged into one point cloud with
    per-point neuron identity, so a frame's colours are one scatter write."""

    def __init__(self, circuit: dict):
        self.points = []
        self.neuron = []      # global neuron index per point
        self.population = []  # population id per point
        self.offsets = {}
        self.counts = {}
        base = 0
        for pop_id, pop in enumerate(("pn", "kc", "mbon", "dan")):
            with np.load(NPZ_DIR / f"{pop}.npz") as d:
                n = len(d["body_id"])
                pos = {int(b): i for i, b in enumerate(d["body_id"])}
                for i, body in enumerate(circuit[f"{pop}_body"]):
                    j = pos.get(int(body))
                    if j is None:
                        continue
                    pts = d[f"pts_{j}"]
                    self.points.append(pts)
                    self.neuron.append(np.full(len(pts), base + i, dtype=np.int64))
                    self.population.append(np.full(len(pts), pop_id, dtype=np.int64))
            self.offsets[pop] = base
            self.counts[pop] = int(len(circuit[f"{pop}_body"]))
            base += self.counts[pop]
        self.points = np.concatenate(self.points)
        self.neuron = np.concatenate(self.neuron)
        self.population = np.concatenate(self.population)
        self.n_neurons = base
        print(f"skeleton cloud: {len(self.points):,} points, {base} neurons")


def population_activity(brain, obs, rpe) -> dict[str, np.ndarray]:
    """One frame of per-neuron activity in [0, 1] for a single-bird obs."""
    o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    pn = brain._pn(brain.encode(o), np.zeros((1, 2), np.float32))[0]
    pn = pn / max(pn.max(), 1e-9)

    kc = brain.kc_all_actions(o)[0, 0].astype(np.float32)  # chosen action's code

    drives = (brain.drives(brain.kc_all_actions(o)))[0]  # (2, n_mbon)
    a = int(np.argmax(brain.values(o)[0]))
    mbon = np.abs(drives[a])
    mbon = mbon / max(mbon.max(), 1e-9)

    # DANs: PAM-innervating neurons signal positive RPE, PPL1 negative RPE
    circuit = brain.circuit
    comp = circuit["comp_class"]
    dan_comp = circuit["dan_comp"]
    pam = (dan_comp[:, comp == 0]).any(axis=1)   # REWARD compartments
    ppl1 = (dan_comp[:, comp == 1]).any(axis=1)  # PUNISHMENT compartments
    dan = np.zeros(dan_comp.shape[0], dtype=np.float32)
    dan[pam] = max(rpe, 0.0)
    dan[ppl1] = max(-rpe, 0.0)
    dan = np.clip(dan, 0, 1) * 0.9 + 0.02

    return {"pn": pn, "kc": kc, "mbon": mbon, "dan": dan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="brain.mp4")
    ap.add_argument("--ticks", type=int, default=600, help="env ticks to record")
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    from .realbrain import RealMB, RealMBConfig

    circuit = load_circuit()
    brain = RealMB(circuit, RealMBConfig(seed=0, alpha=args.alpha))
    brain.load(args.weights)

    cloud = SkeletonCloud(circuit)
    shell = pv.read(PLY_DIR / "JRCFIB2022M_brain.ply")
    shell.points = shell.points / 1000.0  # shell is in nm, skeletons in um
    brain_center = np.array(shell.center, dtype=np.float64)

    # skeletons and shell are now both in MaleCNS micrometre space
    points = cloud.points
    center = points.mean(axis=0)
    print(f"skeleton centroid: {center.round(1)} (shell center {brain_center.round(1)})")

    plotter = pv.Plotter(off_screen=True, window_size=(1280, 800))
    plotter.set_background((0.015, 0.016, 0.025))
    plotter.add_mesh(shell, color=(0.30, 0.38, 0.60), opacity=0.12)

    # lines through consecutive skeleton samples, grouped per neuron
    lines = []
    order = np.lexsort((cloud.population, cloud.neuron))
    pts_sorted = points[order]
    neuron_sorted = cloud.neuron[order]
    start = 0
    for i in range(1, len(neuron_sorted) + 1):
        if i == len(neuron_sorted) or neuron_sorted[i] != neuron_sorted[start]:
            n = i - start
            if n > 1:
                lines.append(np.hstack([[n], np.arange(start, i)]))
            start = i
    lines = np.concatenate(lines)
    poly = pv.PolyData(pts_sorted)
    poly.lines = lines
    rgb = activity_color(np.zeros(cloud.n_neurons)[cloud.neuron])
    poly.point_data["act"] = rgb
    plotter.add_mesh(poly, scalars="act", rgb=True, line_width=1.5,
                     render_lines_as_tubes=False)

    plotter.camera_position = 'iso'
    plotter.camera.focal_point = tuple(brain_center)
    dist = 700.0
    plotter.camera.position = tuple(brain_center + np.array([-500, -300, -400]) * dist / 700)
    plotter.camera.up = (0, -1, 0)
    plotter.camera.clipping_range = (1, 5000)

    plotter.open_movie(args.out, framerate=args.fps)
    orbit_deg = 45.0
    n_expected = max(args.ticks // args.sample_every, 1)
    az_per_frame = orbit_deg / n_expected
    env = FlappyEnv(1, seed=args.seed)
    obs = env.reset()
    prev_q = float(brain.values(obs)[0].max())
    frames = 0
    for tick in range(args.ticks):
        a, q, _ = brain.act(obs, 0.0)
        nobs, r, done = env.step(a)
        qn = float(brain.values(nobs)[0].max()) if not done else 0.0
        rpe = float(r[0]) + brain.cfg.gamma * qn - prev_q
        prev_q = qn
        acts = population_activity(brain, obs, rpe)
        act_vec = np.zeros(cloud.n_neurons, dtype=np.float32)
        for pop, v in acts.items():
            act_vec[cloud.offsets[pop]:cloud.offsets[pop] + cloud.counts[pop]] = v
        poly.point_data["act"] = activity_color(act_vec[cloud.neuron])
        plotter.render()
        if tick % args.sample_every == 0:
            plotter.write_frame()
            plotter.camera.azimuth += az_per_frame
            frames += 1
        obs = nobs
        if done:
            obs = env.reset()
            prev_q = 0.0
    plotter.close()
    print(f"wrote {frames} frames -> {args.out}")
    plotter.screenshot(str(Path(args.out).with_suffix(".png")))


if __name__ == "__main__":
    main()
