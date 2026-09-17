"""Train the mushroom body to play flappy.

Example:
    python -m flybrain_flap.train --minutes 10 --num-envs 64 --out runs/flap1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .brain import MBConfig, MushroomBody
from .env import FlappyEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minutes", type=float, default=10.0)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="discount (episodes are ~200+ ticks)")
    p.add_argument("--eps-start", type=float, default=0.2)
    p.add_argument("--eps-end", type=float, default=0.01)
    p.add_argument("--explore-flap", type=float, default=0.15,
                   help="flap probability while exploring")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default="runs/flap")
    p.add_argument("--circuit", type=str, default=None,
                   help="path to circuit.npz: use the real MaleCNS-connectome "
                        "brain (RealMB) instead of the abstract one")
    p.add_argument("--shaping", type=float, default=0.05,
                   help="gap-alignment shaping reward per step")
    p.add_argument("--trace-lambda", type=float, default=0.0,
                   help="RealMB only: eligibility-trace decay (0 = off, "
                        "~0.97 = synapses stay marked ~30 ticks)")
    p.add_argument("--alpha", type=float, default=0.02,
                   help="RealMB only: target dV per unit RPE")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="RealMB only: target KC-code overlap between actions")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = FlappyEnv(args.num_envs, seed=args.seed, shaping=args.shaping)
    if args.circuit:
        from .connectome import load as load_circuit
        from .realbrain import RealMB, RealMBConfig, calibrate_from_env

        cfg = RealMBConfig(gamma=args.gamma, alpha=args.alpha, seed=args.seed,
                           target_action_overlap=args.overlap,
                           trace_lambda=args.trace_lambda)
        brain = RealMB(load_circuit(args.circuit), cfg)
        info = calibrate_from_env(brain, env)
        print("calibration:", {k: round(v, 4) if isinstance(v, float) else v
                                for k, v in info.items()})
    else:
        cfg = MBConfig(num_features=env.obs().shape[1], eta=args.eta,
                       gamma=args.gamma, seed=args.seed)
        brain = MushroomBody(cfg)

    obs = env.reset()
    eps = args.eps_start
    explore_probs = np.array([1.0 - args.explore_flap, args.explore_flap])
    t0 = time.time()
    next_report = 1.0  # minutes
    minute = 0
    best_pipes = 0
    best_actions: list[int] = []

    episode_pipes: list[float] = []
    trace: list[int] = []  # actions of env 0 in its current episode
    prev_alive = env.alive.copy()

    with open(out / "train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "minute", "episodes", "mean_pipes", "best_pipes", "eps",
        ])

        while (time.time() - t0) / 60.0 < args.minutes:
            actions, q, kc_cache = brain.act(obs, eps, explore_probs=explore_probs)
            trace.append(int(actions[0]))
            next_obs, rewards, done = env.step(actions)

            newly_dead = prev_alive & ~env.alive
            episode_pipes.extend(env.pipes_passed[newly_dead].tolist())

            # greedy bootstrap value of the next state (0 for agents that died)
            q_next = brain.values(next_obs).max(axis=1)
            ar = np.arange(args.num_envs)
            brain.learn(
                actions, kc_cache, rewards, q[ar, actions], q_next,
                dones=newly_dead, alive=prev_alive,
            )
            prev_alive = env.alive.copy()
            obs = next_obs

            if done:
                env0_pipes = int(env.pipes_passed[0])
                if env0_pipes > best_pipes:
                    best_pipes = env0_pipes
                    best_actions = trace
                trace = []
                obs = env.reset()
                prev_alive = env.alive.copy()

            elapsed = (time.time() - t0) / 60.0
            if elapsed >= next_report:
                minute += 1
                eps = max(
                    args.eps_end,
                    args.eps_start
                    - (args.eps_start - args.eps_end) * minute / max(args.minutes, 1e-9),
                )
                episodes = len(episode_pipes)
                mean_pipes = float(np.mean(episode_pipes)) if episode_pipes else 0.0
                best = float(np.max(episode_pipes)) if episode_pipes else 0.0
                writer.writerow([
                    minute, episodes, f"{mean_pipes:.2f}", f"{best:.0f}",
                    f"{eps:.3f}",
                ])
                f.flush()
                print(
                    f"min {minute:3d}  eps {eps:.3f}  episodes {episodes:6d}  "
                    f"mean pipes {mean_pipes:6.2f}  best {best:.0f}",
                    flush=True,
                )
                episode_pipes = []
                next_report += 1.0

    brain.save(str(out / "weights.npz"))
    (out / "best_actions.json").write_text(json.dumps(best_actions))
    print(f"saved weights to {out / 'weights.npz'} (best run: {best_pipes} pipes)")


if __name__ == "__main__":
    main()
