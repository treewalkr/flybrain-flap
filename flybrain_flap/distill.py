"""Expressivity probe: distill the abstract brain's policy into RealMB.

The abstract brain (teacher) flies; the real-connectome brain (student)
learns through its dopamine rule to choose what the teacher would choose.
This separates "can the real substrate EXPRESS a competent policy" from
"can it DISCOVER one by exploration":

- student plays well after distillation  -> substrate is fine; fix training
- student can't even imitate             -> substrate/encoding is the limit

Rewards: env rewards + agreement bonus (+1 teacher's action == student's
greedy action, -1 otherwise). Executed actions are the teacher's.

Example:
    python -m flybrain_flap.distill --minutes 15 --teacher runs/v02/weights.npz \
        --circuit data/circuit.npz --out runs/realprobe
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .brain import MBConfig, MushroomBody
from .env import FlappyEnv


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minutes", type=float, default=15.0)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--teacher", type=str, default="runs/v02/weights.npz")
    p.add_argument("--circuit", type=str, default="data/circuit.npz")
    p.add_argument("--agree", type=float, default=1.0,
                   help="reward for matching the teacher's action")
    p.add_argument("--shaping", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--overlap", type=float, default=0.3)
    p.add_argument("--trace-lambda", type=float, default=0.97)
    p.add_argument("--eval-every", type=float, default=3.0,
                   help="minutes between greedy evaluations")
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default="runs/realprobe")
    return p.parse_args()


def greedy_eval(student, env, episodes: int, seed: int) -> dict:
    """Fly the student alone (no teacher, no exploration) and score it."""
    eval_env = FlappyEnv(env.num_envs, seed=seed, shaping=env.shaping)
    obs = eval_env.reset()
    scores, episodes_done, agree_hits, steps = [], 0, 0, 0
    while episodes_done < episodes:
        a, _, _ = student.act(obs, 0.0)
        obs, _, done = eval_env.step(a)
        steps += 1
        if done:
            scores.extend(eval_env.pipes_passed.tolist())
            episodes_done += eval_env.num_envs
            obs = eval_env.reset()
    return {
        "mean_pipes": float(np.mean(scores)),
        "best": int(np.max(scores)),
        "pass_rate": float(np.mean(np.array(scores) > 0)),
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from .connectome import load as load_circuit
    from .realbrain import RealMB, RealMBConfig, calibrate_from_env

    env = FlappyEnv(args.num_envs, seed=args.seed, shaping=args.shaping)
    student = RealMB(load_circuit(args.circuit),
                     RealMBConfig(gamma=0.99, alpha=args.alpha, seed=args.seed,
                                  target_action_overlap=args.overlap,
                                  trace_lambda=args.trace_lambda))
    calibrate_from_env(student, env)

    teacher = MushroomBody(MBConfig(num_features=env.obs().shape[1]))
    teacher.load(args.teacher)

    obs = env.reset()
    t0 = time.time()
    next_eval = args.eval_every
    n_ar = np.arange(args.num_envs)
    prev_alive = env.alive.copy()
    agree_ema = 0.5
    pending_kcs = None

    with open(out / "distill.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minute", "agree_rate", "student_mean", "student_best",
                    "student_pass", "episodes"])
        print(f"teacher {args.teacher} -> student RealMB "
              f"(lambda={args.trace_lambda})", flush=True)
        while True:
            elapsed = (time.time() - t0) / 60.0
            if elapsed >= args.minutes:
                break
            if elapsed >= next_eval:
                ev = greedy_eval(student, env, args.eval_episodes, seed=99)
                w.writerow([f"{elapsed:.1f}", f"{agree_ema:.3f}",
                            f"{ev['mean_pipes']:.2f}", ev["best"],
                            f"{ev['pass_rate']:.2f}", ""])
                f.flush()
                print(f"min {elapsed:4.1f}  agree {agree_ema:.3f}  "
                      f"student greedy: mean {ev['mean_pipes']:.2f}  "
                      f"best {ev['best']}  pass {ev['pass_rate']:.2f}",
                      flush=True)
                next_eval += args.eval_every

            ta, _, _ = teacher.act(obs, 0.0)             # teacher flies
            sa, q, kc_cache = student.act(obs, 0.0, kcs=pending_kcs)
            agree = (ta == sa)                            # metric only
            agree_ema += 0.001 * (float(agree.mean()) - agree_ema)

            next_obs, rewards, done = env.step(ta)        # execute teacher
            # constant imitation bonus on the EXECUTED action: TD bootstrap
            # raises Q(s, teacher) at visited states, so greedy tracks the
            # teacher. (A disagreement penalty backfires: it depresses the
            # very action we want reinforced when the student strays.)
            rewards = rewards + args.agree

            newly_dead = prev_alive & ~env.alive
            pending_kcs = student.kc_all_actions(next_obs)
            q_next = student._q_from_kc(pending_kcs).max(axis=1)
            student.learn(ta, kc_cache, rewards, q[n_ar, ta], q_next,
                          dones=newly_dead, alive=prev_alive)
            prev_alive = env.alive.copy()
            obs = next_obs if not done else env.reset()
            prev_alive = env.alive.copy() if done else prev_alive
            if done:
                pending_kcs = None

    ev = greedy_eval(student, env, args.eval_episodes, seed=99)
    print("final:", ev, flush=True)
    student.save(str(out / "weights.npz"))
    print(f"saved weights to {out / 'weights.npz'}", flush=True)


if __name__ == "__main__":
    main()
