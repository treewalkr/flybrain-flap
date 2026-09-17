# flybrain-flap

A fruit fly brain plays a Flappy-Bird-like game.

Inspired by [TMNF-C](https://github.com/adonis-singh/TMNF-C), which wired
the MaleCNS *Drosophila* connectome into a TrackMania simulator.

A lightweight flappy environment (pure NumPy, vectorized) driven by a
mushroom body modeled on the *Drosophila* learning circuit: projection
neurons encode the state, Kenyon cells form a sparse coincidence code,
mushroom body output neurons read that code, and dopamine
reward-prediction error depresses and restores the active synapses.
No gradients, no backprop — the fly's own trial-and-error rule.
The circuit is small on purpose: 35 PNs, 1,024 KCs, 16 MBONs.

## Setup

Python 3.10+ with NumPy. For the graphical game, pygame-ce:

```bash
python -m venv .venv && .venv/bin/pip install numpy pygame-ce pytest
```

Everything runs from the repo root without installing the package.

## Quick start

```bash
# play it yourself first (SPACE or click to flap)
.venv/bin/python -m flybrain_flap.game --human

# train the brain for 10 minutes on 64 parallel birds
.venv/bin/python -m flybrain_flap.train --minutes 10 --num-envs 64 \
    --eta 0.05 --gamma 0.99 --out runs/flap1

# watch the trained brain play in realtime, with the live
# mushroom-body activity panel (PNs, KC codes, MBONs, values, dopamine)
.venv/bin/python -m flybrain_flap.game --brain runs/flap1/weights.npz --brain-view

# or headless, as ASCII frames
.venv/bin/python -m flybrain_flap.play runs/flap1/weights.npz --episodes 3
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
