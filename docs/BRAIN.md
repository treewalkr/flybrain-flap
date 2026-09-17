# The brain

A simplified mushroom body: the *Drosophila* circuit that learns to
approach or avoid odours by dopamine-mediated plasticity. Here the
"odour" is the game state, and approach/avoid becomes good/bad action
values.

## Wiring

```
 state (5) + one-hot action (2)      fixed random sparse
            │                        ┌────────────────┐
            ▼                        ▼                │
     Projection neurons (PN) ──►  Kenyon cells (KC)  │  top-k winner-take-all
            7                  512 (20 active)        │
                                       │            │
                                       ▼            │
                              Mushroom body output neurons (MBON)
                                     16 total: 8 approach, 8 avoid
```

- **PN → KC**: fixed random binary matrix, ~6 PNs per KC, never learned.
- **KC firing**: only the 20 KCs with the largest drive fire — a sparse,
  decorrelated code of (state, action). Sparse codes make synapse
  updates touch only a few weights, like the real ~5% KC response.
- **KC → MBON**: learned weights, bounded in [0, w0].

## Value and action

The value of an action is the summed drive of its active KC synapses
onto approach MBONs minus the drive onto avoid MBONs:

    Q(s, a) = Σ w(approach) − Σ w(avoid)

Greedy policy with epsilon-greedy exploration.

## The dopamine rule

The reward-prediction error

    RPE = r + γ·Q(s', a') − Q(s, a)

becomes dopamine, split by compartment (Handler et al. 2019,
Bennett et al. 2021):

- **Positive RPE → PAM compartments** → depress active KC→*avoid*
  synapses (the outcome was better than expected, so unlearn avoidance).
- **Negative RPE → PPL1 compartments** → depress active KC→*approach*
  synapses (unlearn approach).

Dopamine only ever *depresses* active synapses. Weights recover toward
w0 slowly when the synapse is active without dopamine. Everything stays
in [0, w0]. No gradients anywhere.

## Known simplifications

- State and action enter the PNs directly rather than through antennal
  lobe glomeruli.
- No lateral inhibition / feed-forward normalization between PN and KC
  beyond the top-k.
- One compartment per MBON class rather than the 15 real lobes.
- Rate-based; no spiking.

The plan is to make each of these more realistic over time — see
[ROADMAP.md](ROADMAP.md).
