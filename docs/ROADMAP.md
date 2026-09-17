# Roadmap

Status markers: [ ] todo, [~] in progress, [x] done.

## v0.1 — the loop works

- [x] Vectorized flappy environment (N birds, fixed-tick physics)
- [x] Mushroom body: PN tuning curves → KC coincidence code → MBON,
      dopamine RPE rule with recovery
- [x] Per-action KC pools so action values are read from disjoint codes
- [x] Training loop with per-minute logging and weight checkpoints
- [x] Human-playable pygame game
- [x] Realtime brain playback in the same game
- [x] Greedy evaluation + ASCII playback
- [~] Birds reliably clear pipes after a short training run

## v0.2 — make the fly more real

- [ ] Mixed KC code read out by competing MBONs (drop the per-action
      pools; actions bias the MBON population instead)
- [ ] PN gain/threshold calibration from a random-policy rollout
- [ ] Feed-forward inhibition onto KCs instead of plain top-k
- [ ] Eligible-trace gating so dopamine only hits recently active synapses
- [ ] Multiple dopamine compartments with different rates

## v0.3 — see like a fly

- [ ] Render the game to a low-res retinal image; hexagonal ommatidia
      sampling replaces the state features as observation
- [ ] A looming detector (LC4-like) as PN input

## v0.4 — a real connectome

- [ ] Load real PN→KC and KC→MBON synapse counts from a published
      connectome table instead of random wiring
- [ ] Pathway audit script verifying wiring facts against the raw tables
- [ ] Visualize KC/MBON/DAN activity during a trained run

## Stretch

- [ ] Web viewer for runs (game canvas + activity traces)
- [ ] Swap the game: same brain, different env, zero brain changes
