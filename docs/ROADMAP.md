# Roadmap

Status markers: [ ] todo, [~] in progress, [x] done.

## v0.1 — the loop works

- [x] Vectorized flappy environment (N birds, fixed-tick physics)
- [x] Mushroom body: PN → KC (sparse, top-k) → MBON, dopamine RPE rule
- [x] Training loop with per-minute logging and weight checkpoints
- [x] Greedy evaluation + ASCII playback
- [x] Birds reliably clear pipes after a short training run

## v0.2 — make the fly more real

- [ ] PN gain/threshold calibration from a random-policy rollout
      (activity statistics measured once, never learned)
- [ ] Feed-forward inhibition onto KCs (AKAP-like) instead of plain top-k
- [ ] Eligible-trace gating so dopamine only hits recently active synapses
- [ ] Separate dopamine "types" per MBON compartment with different rates

## v0.3 — see like a fly

- [ ] Render the game to a low-res retinal image; hexagonal ommatidia
      sampling replaces the state features as observation
- [ ] A simple optic-flow feature (looming detector) as PN input

## v0.4 — a real connectome

- [ ] Load real PN→KC and KC→MBON synapse counts from a published
      connectome table instead of random projections
- [ ] Pathway audit script verifying wiring facts against the raw tables
- [ ] Visualize KC/MBON/DAN activity during a trained run

## Stretch

- [ ] Web viewer for runs (game canvas + activity traces)
- [ ] Swap the game: same brain, different env, zero brain changes
