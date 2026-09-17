# The brain

A simplified mushroom body: the *Drosophila* circuit that learns to
approach or avoid odours by dopamine-mediated plasticity. Here the
"odour" is the game state, and approach/avoid becomes good/bad action
values.

## Wiring

```
 state features (5)                PN tuning curves (7 bins/feature)
        │                                   │
        ▼                                   ▼
   ┌────────────┐   random sparse    ┌─────────────┐
   │ Projection │ ────────────────► │ Kenyon cells │  coincidence code:
   │  neurons   │   ~3 PNs per KC   │  2×512 pools │  top-16 per pool
   └────────────┘                   └──────┬──────┘
                                           │
                                           ▼
                          Mushroom body output neurons (MBON)
                          16 total: 8 approach + 8 avoid,
                          each reading all KCs
```

- **PNs** are feature detectors with broad triangular tuning curves
  (7 bins per feature): each feature softly activates 1–2 PNs, like
  odour glomeruli responding to odorant concentration.
- **KCs** are coincidence detectors: each Kenyon cell reads ~3 random
  PNs (from distinct features) and is driven by the *weakest* of them —
  a soft AND. Only the 16 KCs with the strongest coincidence fire per
  pool (~3% sparse code, like the real ~5%).
- **KC pools**: Kenyon cells are split into one pool per candidate
  action, so the (state, action) code is disjoint between actions and
  each action's value is read from its own synapses. This stands in for
  the real circuit's compartment-specific output neurons.
- **KC → MBON**: learned weights in [0, w0], starting fully potentiated.
  The 16 MBONs are shared: each reads all 1,024 KCs, so only the active
  pool's synapses contribute to a given action's value.
- **Dopamine neurons** are not explicit nodes: the RPE is a broadcast
  signal applied compartment-wise (the fly has ~340 DANs).

## Value and action

The value of an action is the summed drive of its pool's active KC
synapses onto approach MBONs minus the drive onto avoid MBONs,
normalized to roughly [-1, 1]:

    Q(s, a) = (Σ w(approach) − Σ w(avoid)) / (kc_active × mbon)

Greedy policy with epsilon-greedy exploration (biased toward gliding
via `--explore-flap`, since random flapping is quickly lethal in this
game). Discount γ = 0.99 in training: episodes run hundreds of ticks
and a pipe passes roughly every 84 of them.

## The dopamine rule

The reward-prediction error

    RPE = r + γ·Q(s', a') − Q(s, a)

becomes dopamine, split by compartment (Handler et al. 2019,
Bennett et al. 2021):

- **Above-baseline dopamine (positive RPE) → PAM neurons** depress the
  active KC→*avoid* synapses: the outcome was better than expected, so
  unlearn avoidance. Below-baseline dopamine lets them recover.
- **Below-baseline dopamine (negative RPE) → PPL1 neurons** depress the
  active KC→*approach* synapses and recovery runs the other way.

So deviations from the dopamine baseline drive depression *and*
recovery symmetrically, weights stay in [0, w0], and no gradients are
involved anywhere. Weight updates are averaged over the batch of birds.

## Known simplifications

- State features enter PNs directly rather than through a full antennal
  lobe with lateral inhibition.
- One compartment class per MBON rather than the 15 real lobes.
- Per-action KC pools instead of a single mixed code read out by
  competing MBONs (the mixed code is on the roadmap).
- Rate-based; no spiking.

The plan is to make each of these more realistic over time — see
[ROADMAP.md](ROADMAP.md).

## References and inspiration

- This project is a small-scale homage to
  [TMNF-C](https://github.com/adonis-singh/TMNF-C), which wired the
  MaleCNS *Drosophila* connectome into a TrackMania simulator — the
  dopamine-rule formulation and the honest reporting of what a mushroom
  body can and cannot learn both follow its example.
- Handler, A. et al. (2019), Cell; Bennett, J. E. M. et al. (2021),
  Nature Communications — dopamine-mediated synaptic depression and
  recovery in the mushroom body.
- Aso, Y. et al. (2014); Hige, T. et al. (2015) — the compartment
  organization this circuit simplifies.
