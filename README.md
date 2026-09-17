# flybrain-flap

A fruit fly brain plays a Flappy-Bird-like game.

A lightweight flappy environment (pure NumPy, vectorized) driven by a
mushroom body inspired by the *Drosophila* learning circuit: projection
neurons encode the state, Kenyon cells form a sparse coincidence code,
mushroom body output neurons read that code, and dopamine
reward-prediction error depresses and restores the active synapses.
No gradients, no backprop — the fly's own trial-and-error rule.

## Setup

Python 3.10+ with NumPy. For the graphical game, pygame-ce:

```bash
python -m venv .venv && .venv/bin/pip install numpy pygame-ce pytest
```

Everything runs from the repo root without installing the package.

## Quick start

```bash
# play it yourself first (SPACE or click to flap)
python -m flybrain_flap.game --human

# train the brain for 10 minutes on 64 parallel birds
python -m flybrain_flap.train --minutes 10 --num-envs 64 --out runs/flap1

# watch the trained brain play in realtime
python -m flybrain_flap.game --brain runs/flap1/weights.npz

# or headless, as ASCII frames
python -m flybrain_flap.play runs/flap1/weights.npz --episodes 3
```

Training writes `train.csv` (per-minute stats), `weights.npz` (the
learned KC→MBON synapses) and `best_actions.json` (the best run's
action trace).

## Docs

- [Architecture](docs/ARCHITECTURE.md) — components and data flow
- [Brain](docs/BRAIN.md) — the mushroom body circuit and the dopamine rule
- [Roadmap](docs/ROADMAP.md) — where this is going

## License

MIT
