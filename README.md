# flybrain-flap

A fruit fly brain plays a Flappy-Bird-like game.

A lightweight flappy environment (pure NumPy, vectorized) driven by a
mushroom body inspired by the *Drosophila* learning circuit: Kenyon cells
form a sparse code of (state, action), mushroom body output neurons read
that code, and dopamine reward-prediction error depresses active
Kenyon-cell synapses. No gradients, no backprop — the fly's own
trial-and-error rule.

## Install

```bash
pip install -e .
```

## Quick start

```bash
# train for 10 minutes on 64 parallel birds
python -m flybrain_flap.train --minutes 10 --num-envs 64 --out runs/flap1

# watch the trained policy (ASCII render)
python -m flybrain_flap.play runs/flap1/weights.npz --episodes 3
```

## Docs

- [Architecture](docs/ARCHITECTURE.md) — components and data flow
- [Brain](docs/BRAIN.md) — the mushroom body circuit and the dopamine rule
- [Roadmap](docs/ROADMAP.md) — where this is going

## License

MIT
