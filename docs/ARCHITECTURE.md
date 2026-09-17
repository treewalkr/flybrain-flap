# Architecture

Three components, deliberately decoupled so each can be swapped:

```
            observations (N, 5)
   ┌─────────────────────────────►  brain (mushroom body)
   │                                        │
   │                                 actions (N,) in {0,1}
   │                                        │
   ┌── env (vectorized flappy) ◄────────────┘
   │                                        │
   └── rewards, dones ────────────────►  brain.learn()
```

## Environment (`env.py`)

`FlappyEnv` runs N birds in parallel through the same pipe sequence, one
NumPy array per state field. Physics is a fixed tick: gravity accelerates
vertical velocity, the flap action sets it upward, pipes scroll left.
The parameters are tuned so a mostly-gliding bird hovers near balance
(flap ≈ every 9 ticks), which keeps random exploration alive long
enough to reach pipes.

Each bird tracks its own episode; dead birds are held frozen (observation
zeroed, zero reward) until all N are done, then the whole batch resets.

Observation (all normalized to roughly [-1, 1]):

| # | feature |
|---|---------|
| 0 | bird height |
| 1 | bird vertical velocity |
| 2 | distance to next pipe |
| 3 | relative height of the gap center |
| 4 | half gap size |

Rewards: +1.0 per pipe passed, +0.01 per step survived, +0.05 scaled by
closeness to the gap center height (dense shaping so the dopamine signal
is informative before the first pipe is ever passed), -1.0 on death.

## Brain (`brain.py`)

`MushroomBody` maps observation + candidate action to a value and picks
the best action. Learning happens through a dopamine reward-prediction
error rule — see [BRAIN.md](BRAIN.md).

## Training loop (`train.py`)

Standard actor loop: step env, learn from cached forward passes,
epsilon-greedy with decay, per-minute CSV log, best-run action trace,
weights saved as `.npz`. Runs from the repo root with no install:

```bash
python -m flybrain_flap.train --minutes 10 --num-envs 64 --out runs/flap1
```

## Game (`game.py`)

Pygame rendering of a single bird, two modes:

- `python -m flybrain_flap.game --human` — you play (SPACE / click to
  flap, P pause, R restart)
- `python -m flybrain_flap.game --brain runs/flap1/weights.npz` — watch
  the trained brain play in realtime (SPACE speeds up, auto-restarts
  after game over)

## Playing headless (`play.py`)

Loads saved weights, runs the env one bird at a time with greedy actions,
prints ASCII frames. No learning, no dependencies beyond NumPy.
