# Architecture

Three components, deliberately decoupled so each can be swapped:

```
            observations (N, obs_dim)
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

Each bird tracks its own episode; dead birds are held frozen (observation
zeroed) until all N are done, then the whole batch resets. This keeps the
training loop branch-free.

Observation (all normalized to roughly [-1, 1]):

| # | feature |
|---|---------|
| 0 | bird height |
| 1 | bird vertical velocity |
| 2 | distance to next pipe |
| 3 | relative height of the gap center |
| 4 | half gap size |

## Brain (`brain.py`)

`MushroomBody` maps observation + candidate action to a value and picks
the best action. Learning happens through a dopamine reward-prediction
error rule — see [BRAIN.md](BRAIN.md).

## Training loop (`train.py`)

Standard actor loop: step env, learn from (s, a, r, s'), epsilon-greedy
with decay, per-minute CSV log, best-run action trace, weights saved as
`.npz`.

## Playing (`play.py`)

Loads saved weights, runs the env one bird at a time with greedy actions,
prints ASCII frames. No learning.
