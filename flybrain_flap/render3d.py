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
from collections import deque
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


def population_activity(brain, obs, rpe, kcs=None) -> dict[str, np.ndarray]:
    """One frame of per-neuron activity in [0, 1] for a single-bird obs.
    kcs: precomputed (1, 2, n_kc) codes from brain.kc_all_actions (optional)."""
    o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    pn = brain._pn(brain.encode(o), np.zeros((1, 2), np.float32))[0]
    pn = pn / max(pn.max(), 1e-9)

    kcs = brain.kc_all_actions(o) if kcs is None else kcs
    kc = kcs[0, 0].astype(np.float32)  # chosen action's code
    drives = brain.drives(kcs)[0]      # (2, n_mbon)
    a = int(np.argmax(brain._q_from_kc(kcs)[0]))
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


def build_scene(cloud: SkeletonCloud, stride: int = 1, off_screen: bool = True,
                window_size=(1280, 800)):
    """One pyvista plotter with the shell + all skeletons as a colourable mesh."""
    import pyvista as pv

    shell = pv.read(PLY_DIR / "JRCFIB2022M_brain.ply")
    shell.points = shell.points / 1000.0  # shell is in nm, skeletons in um
    brain_center = np.array(shell.center, dtype=np.float64)

    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.set_background((0.015, 0.016, 0.025))
    plotter.add_mesh(shell, color=(0.30, 0.38, 0.60), opacity=0.12)

    # lines through consecutive skeleton samples, grouped per neuron
    order = np.lexsort((cloud.population, cloud.neuron))
    keep = order[::stride]  # subsample long skeletons for speed
    pts_sorted = cloud.points[keep]
    neuron_sorted = cloud.neuron[keep]
    lines = []
    start = 0
    for i in range(1, len(neuron_sorted) + 1):
        if i == len(neuron_sorted) or neuron_sorted[i] != neuron_sorted[start]:
            if i - start > 1:
                lines.append(np.hstack([[i - start], np.arange(start, i)]))
            start = i
    poly = pv.PolyData(pts_sorted)
    poly.lines = np.concatenate(lines)
    poly.point_data["act"] = activity_color(np.zeros(cloud.n_neurons)[cloud.neuron])
    plotter.add_mesh(poly, scalars="act", rgb=True, line_width=2.0)

    plotter.camera_position = 'iso'
    plotter.camera.focal_point = tuple(brain_center)
    plotter.camera.position = tuple(brain_center + np.array([-500, -300, -400]))
    plotter.camera.up = (0, -1, 0)
    plotter.camera.clipping_range = (1, 5000)
    return plotter, poly


def update_colors(poly, cloud: SkeletonCloud, act_vec: np.ndarray):
    poly.point_data["act"] = activity_color(act_vec[cloud.neuron])


class TracePanel:
    """Rolling activity traces beside the game (the reference repo's video
    overlay style): MBON value, dopamine PAM+ / PPL1-, bird height."""

    WIDTH, HEIGHT = 480, 640
    BG = (16, 16, 22)
    TEXT = (200, 200, 210)
    GRID = (52, 56, 66)
    COL_VALUE = (255, 214, 64)
    COL_PAM = (80, 200, 120)
    COL_PPL1 = (220, 90, 90)
    COL_HEIGHT = (120, 170, 255)

    def __init__(self, maxlen: int = 360):
        import pygame

        self.pygame = pygame
        self.font = pygame.font.SysFont(None, 20)
        self.n = maxlen
        self.value = deque(maxlen=maxlen)
        self.pam = deque(maxlen=maxlen)     # +RPE dopamine
        self.ppl1 = deque(maxlen=maxlen)    # -RPE dopamine
        self.height = deque(maxlen=maxlen)

    def push(self, value: float, rpe: float, height: float) -> None:
        self.value.append(value)
        self.pam.append(max(rpe, 0.0))
        self.ppl1.append(max(-rpe, 0.0))
        self.height.append(height)

    def _series(self, surf, box, series, color, label, zero_line=False):
        x0, y0, x1, y1 = box
        self.pygame.draw.rect(surf, (22, 22, 30), box, 1)
        lab = self.font.render(label, True, self.TEXT)
        surf.blit(lab, (x0 + 6, y0 + 2))
        pts = list(series)
        if len(pts) < 2:
            return
        vals = np.array(pts)
        span = max(np.abs(vals).max(), 1e-6)
        if zero_line:
            mid = y0 + (y1 - y0) // 2 + 8
            self.pygame.draw.line(surf, self.GRID, (x0 + 4, mid), (x1 - 4, mid), 1)
            step = (x1 - x0 - 8) / (self.n - 1)
            offset = self.n - len(pts)
            pixels = [
                (x0 + 4 + (offset + i) * step,
                 mid - v / span * ((y1 - y0) // 2 - 14))
                for i, v in enumerate(pts)
            ]
        else:
            lo, hi = float(vals.min()), float(vals.max())
            rng = max(hi - lo, 1e-6)
            step = (x1 - x0 - 8) / (self.n - 1)
            offset = self.n - len(pts)
            pixels = [
                (x0 + 4 + (offset + i) * step,
                 y1 - 14 - (v - lo) / rng * (y1 - y0 - 30))
                for i, v in enumerate(pts)
            ]
        self.pygame.draw.lines(surf, color, False, pixels, 2)

    def draw(self, surf) -> None:
        self.pygame.draw.rect(surf, self.BG, (0, 0, self.WIDTH, self.HEIGHT))
        w = self.WIDTH - 24
        h = self.HEIGHT
        self._series(surf, (12, 10, 12 + w, 10 + int(0.22 * h)),
                     self.value, self.COL_VALUE,
                     "MBON value  (approach - avoid)", zero_line=True)
        self._series(surf, (12, int(0.25 * h), 12 + w, int(0.25 * h) + int(0.24 * h)),
                     self.pam, self.COL_PAM,
                     "dopamine  PAM + (better than expected)")
        self._series(surf, (12, int(0.52 * h), 12 + w, int(0.52 * h) + int(0.24 * h)),
                     self.ppl1, self.COL_PPL1,
                     "dopamine  PPL1 - (worse than expected)")
        self._series(surf, (12, int(0.79 * h), 12 + w, int(0.79 * h) + int(0.19 * h)),
                     self.height, self.COL_HEIGHT,
                     "bird height")


def run_live(args) -> None:
    """Real-time: pygame game window + interactive 3D brain window, one loop."""
    import pygame
    from collections import deque

    from .game import SCREEN_W, GameView
    from .realbrain import RealMB, RealMBConfig

    circuit = load_circuit()
    brain = RealMB(circuit, RealMBConfig(seed=0, alpha=args.alpha))
    brain.load(args.weights)

    cloud = SkeletonCloud(circuit)
    print(f"subsampling skeletons 1/{args.stride} for interactive speed")
    cloud.points = cloud.points[::args.stride]
    cloud.neuron = cloud.neuron[::args.stride]
    cloud.population = cloud.population[::args.stride]

    plotter, poly = build_scene(cloud, stride=1, off_screen=False)
    plotter.show(interactive_update=True, auto_close=False)

    view = GameView(seed=args.seed, brain_view=True)
    traces = TracePanel()
    obs = view.env.reset()
    prev_q = float(brain.values(obs)[0].max())
    clock = view.clock
    speed = 1
    running = True
    frame = 0
    while running:
        for event in view.pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    speed = min(speed * 2, 8)
                elif event.key == pygame.K_r:
                    obs = view.env.reset()
                    prev_q = 0.0

        done = False
        rpe = None
        q_taken = 0.0
        kcs = None
        for _ in range(speed):
            a, q, kc_list = brain.act(obs, 0.0)
            q_taken = float(q[0, a[0]])
            kcs = np.stack(kc_list, axis=0)[None]  # (1, 2, n_kc), cached
            nobs, r, done = view.env.step(a)
            qn = float(brain.values(nobs)[0].max()) if not done else 0.0
            rpe = float(r[0]) + brain.cfg.gamma * qn - prev_q
            prev_q = qn
            obs = nobs
            if done:
                obs = view.env.reset()
                prev_q = 0.0
        view.draw(flash_flap=bool(a[0] == 1))
        traces.push(q_taken, rpe if rpe is not None else 0.0, float(obs[0][0]))
        traces.draw(view.screen.subsurface((SCREEN_W, 0, 480, view.screen.get_height())))

        frame += 1
        if frame % 2 == 0:  # 3D at ~30 fps: the framerate bottleneck is VTK
            acts = population_activity(brain, obs, rpe if rpe is not None else 0.0, kcs=kcs)
            act_vec = np.zeros(cloud.n_neurons, dtype=np.float32)
            for pop, v in acts.items():
                act_vec[cloud.offsets[pop]:cloud.offsets[pop] + cloud.counts[pop]] = v
            update_colors(poly, cloud, act_vec)
            plotter.render()
        clock.tick(60)

    plotter.close()
    view.pygame.quit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="brain.mp4")
    ap.add_argument("--ticks", type=int, default=600, help="env ticks to record")
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--live", action="store_true",
                    help="real-time interactive window instead of MP4")
    ap.add_argument("--stride", type=int, default=4,
                    help="skeleton point subsampling in --live mode")
    args = ap.parse_args()

    if args.live:
        run_live(args)
        return

    import pyvista as pv
    pv.OFF_SCREEN = True

    from .realbrain import RealMB, RealMBConfig

    circuit = load_circuit()
    brain = RealMB(circuit, RealMBConfig(seed=0, alpha=args.alpha))
    brain.load(args.weights)

    cloud = SkeletonCloud(circuit)
    brain_center = None

    plotter, poly = build_scene(cloud)

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
