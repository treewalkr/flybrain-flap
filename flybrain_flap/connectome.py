"""Extract the mushroom-body circuit from the MaleCNS v1.0 flat connectome.

    .venv/bin/python -m flybrain_flap.connectome            # build data/circuit.npz
    .venv/bin/python -m flybrain_flap.connectome --report   # audit facts

Circuit contents (data/circuit.npz):
  ALPN -> KC synapse counts (the fixed expansion), KC -> MBON synapse counts
  (the only plastic synapses), MBON compartments parsed from instance names,
  DAN compartments from names or inferred from DAN -> MBON synapses,
  compartment class from the DAN type (PAM = reward, PPL1 = punishment).
  MBONs in reward compartments form the "avoid" group, MBONs in punishment
  compartments the "approach" group (Aso et al. 2014), cross-checked against
  neurotransmitter (glutamate = avoid). Per-neuron skeleton centroids give
  every node a real 3D position for rendering.

Data: MaleCNS v1.0 (Janelia FlyEM), CC BY 4.0. Fetch with
`python -m flybrain_flap.fetch_connectome` first.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .fetch_connectome import MALECNS, NPZ_DIR, CLASSES

CIRCUIT = Path("data/circuit.npz")

REWARD = 0      # PAM compartment: MBONs promote avoidance, depressed by positive RPE
PUNISHMENT = 1  # PPL1 compartment: MBONs promote approach, depressed by negative RPE
APPROACH = 0
AVOID = 1
EXCLUDED = -1

_LOBE = {"y": "g", "a": "a", "B": "b", "b": "b"}
# minimum evidence to infer an unnamed DAN's compartment from its outputs
_INFER_MIN_KC_SYNAPSES = 100
_INFER_MIN_MBON_SYNAPSES = 50


def parse_compartments(instance: str) -> list[str]:
    """``MBON01(y5B'2a)_R`` -> ['g5', "b'2"]; dendritic part before '>' / '<'."""
    match = re.search(r"\(([^)]*)\)", instance)
    if match is None:
        return []
    text = re.split(r"[<>]", match.group(1))[0]
    result: list[str] = []
    lobe = None
    prime = ""
    for i, ch in enumerate(text):
        if ch in _LOBE:
            lobe = _LOBE[ch]
            prime = "'" if i + 1 < len(text) and text[i + 1] == "'" else ""
        elif ch.isdigit() and lobe is not None:
            name = f"{lobe}{prime}{ch}"
            if name not in result:
                result.append(name)
    return result


def _population(ann, cls: str) -> "np.ndarray[tuple]":  # structured rows
    frame = ann[ann["class"] == cls][["bodyId", "type", "instance"]].copy()
    frame["type"] = frame["type"].fillna("").astype(str)
    frame["instance"] = frame["instance"].fillna("").astype(str)
    return frame.sort_values("bodyId").reset_index(drop=True)


def _centroids(pop: str, bodies: np.ndarray) -> np.ndarray:
    """Skeleton centroid (um, xyz) per body; zeros where no skeleton exists."""
    out = np.zeros((len(bodies), 3), dtype=np.float32)
    npz = NPZ_DIR / f"{pop}.npz"
    if not npz.exists():
        return out
    with np.load(npz) as d:
        pos = {int(b): i for i, b in enumerate(d["body_id"])}
        for i, b in enumerate(bodies):
            j = pos.get(int(b))
            if j is not None:
                out[i] = d[f"pts_{j}"].mean(axis=0)
    return out


def extract(out: Path = CIRCUIT) -> dict:
    import pandas as pd

    ann = pd.read_feather(MALECNS / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ann = ann[ann["status"] == "Traced"]
    nt = pd.read_feather(MALECNS / "body-neurotransmitters-male-cns-v1.0.feather")
    nt = nt.set_index("body")["consensus_nt"]

    pops = {name: _population(ann, cls) for name, cls in CLASSES.items()}
    dan = pops["dan"]
    dan = dan[dan["type"].str.startswith(("PAM", "PPL1"))].reset_index(drop=True)
    pops["dan"] = dan
    index = {name: {int(b): i for i, b in enumerate(f["bodyId"])}
             for name, f in pops.items()}

    weights = pd.read_feather(MALECNS / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    all_ids = set().union(*(set(ix) for ix in index.values()))
    weights = weights[weights["body_pre"].isin(all_ids) & weights["body_post"].isin(all_ids)]

    def matrix(pre: str, post: str) -> np.ndarray:
        m = np.zeros((len(index[pre]), len(index[post])), dtype=np.float32)
        sub = weights[weights["body_pre"].isin(index[pre])
                      & weights["body_post"].isin(index[post])]
        m[sub["body_pre"].map(index[pre]).to_numpy(),
          sub["body_post"].map(index[post]).to_numpy()] = sub["weight"].to_numpy(np.float32)
        return m

    pn_kc = matrix("pn", "kc")
    kc_mbon = matrix("kc", "mbon")
    dan_kc = matrix("dan", "kc")
    dan_mbon = matrix("dan", "mbon")

    # MBON compartments from instance names
    mbon_comps = [parse_compartments(s) for s in pops["mbon"]["instance"]]
    # DAN compartments: names first, otherwise inferred from the MBON the DAN
    # synapses onto most (only DANs that actually innervate KCs)
    dan_comps: list[list[str]] = []
    dan_inferred = np.zeros(len(dan), dtype=bool)
    for i, inst in enumerate(dan["instance"]):
        comps = parse_compartments(inst)
        if not comps:
            if dan_kc[i].sum() >= _INFER_MIN_KC_SYNAPSES and dan_mbon[i].max() >= _INFER_MIN_MBON_SYNAPSES:
                comps = list(mbon_comps[int(dan_mbon[i].argmax())])
                dan_inferred[i] = bool(comps)
        dan_comps.append(comps)

    names = sorted({c for comps in mbon_comps + dan_comps for c in comps})
    comp_index = {c: i for i, c in enumerate(names)}

    comp_class = np.full(len(names), EXCLUDED, dtype=np.int64)
    dan_comp = np.zeros((len(dan), len(names)), dtype=bool)
    for i, comps in enumerate(dan_comps):
        if not comps:
            continue
        cls = REWARD if dan["type"][i].startswith("PAM") else PUNISHMENT
        for name in comps:
            c = comp_index[name]
            dan_comp[i, c] = True
            if comp_class[c] not in (EXCLUDED, cls):
                raise ValueError(f"compartment {name} receives both PAM and PPL1 DANs")
            comp_class[c] = cls

    mbon_comp = np.zeros((len(pops["mbon"]), len(names)), dtype=bool)
    mbon_group = np.full(len(pops["mbon"]), EXCLUDED, dtype=np.int64)
    for i, comps in enumerate(mbon_comps):
        classes = set()
        for c in comps:
            mbon_comp[i, comp_index[c]] = True
            classes.add(int(comp_class[comp_index[c]]))
        classes.discard(EXCLUDED)
        if len(classes) == 1:
            mbon_group[i] = AVOID if classes.pop() == REWARD else APPROACH

    mbon_nt = np.array([str(nt.get(int(b), "")) for b in pops["mbon"]["bodyId"]])

    result = {
        "pn_body": pops["pn"]["bodyId"].to_numpy(np.int64),
        "pn_type": pops["pn"]["type"].to_numpy(str),
        "kc_body": pops["kc"]["bodyId"].to_numpy(np.int64),
        "kc_type": pops["kc"]["type"].to_numpy(str),
        "mbon_body": pops["mbon"]["bodyId"].to_numpy(np.int64),
        "mbon_type": pops["mbon"]["type"].to_numpy(str),
        "mbon_instance": pops["mbon"]["instance"].to_numpy(str),
        "mbon_nt": mbon_nt,
        "mbon_comp": mbon_comp,
        "mbon_group": mbon_group,
        "dan_body": dan["bodyId"].to_numpy(np.int64),
        "dan_type": dan["type"].to_numpy(str),
        "dan_comp": dan_comp,
        "dan_inferred": dan_inferred,
        "comp_names": np.array(names),
        "comp_class": comp_class,
        "pn_kc": pn_kc,
        "kc_mbon": kc_mbon,
        "dan_mbon": dan_mbon,
        "pn_pos": _centroids("pn", pops["pn"]["bodyId"].to_numpy(np.int64)),
        "kc_pos": _centroids("kc", pops["kc"]["bodyId"].to_numpy(np.int64)),
        "mbon_pos": _centroids("mbon", pops["mbon"]["bodyId"].to_numpy(np.int64)),
        "dan_pos": _centroids("dan", dan["bodyId"].to_numpy(np.int64)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **result)
    return result


def load(path: Path = CIRCUIT) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def report(c: dict) -> str:
    pn_kc, kc_mbon = c["pn_kc"], c["kc_mbon"]
    kc_in = (pn_kc > 0).sum(axis=0)
    lines = [
        f"PNs (ALPN): {len(c['pn_body'])}, with KC output: {int((pn_kc.sum(1) > 0).sum())}",
        f"KCs: {len(c['kc_body'])}, with PN input: {int((kc_in > 0).sum())}, "
        f"PNs/KC (with input): mean {kc_in[kc_in > 0].mean():.2f} "
        f"median {np.median(kc_in[kc_in > 0]):.0f}",
        f"PN->KC edges {int((pn_kc > 0).sum())}, synapses {int(pn_kc.sum())}",
        f"KC->MBON edges {int((kc_mbon > 0).sum())}, synapses {int(kc_mbon.sum())}",
        f"MBONs: {len(c['mbon_body'])}, DANs: {len(c['dan_body'])}, "
        f"compartments: {len(c['comp_names'])}",
    ]
    for g, label in ((AVOID, "avoid (PAM compartments)"),
                     (APPROACH, "approach (PPL1 compartments)"), (EXCLUDED, "excluded")):
        m = np.flatnonzero(c["mbon_group"] == g)
        types = sorted({str(t) for t in c["mbon_type"][m]})
        lines.append(f"MBON group {label}: {len(m)} neurons, types {types}")
    nt_group = np.where(c["mbon_nt"] == "glutamate", AVOID, APPROACH)
    used = c["mbon_group"] != EXCLUDED
    agree = int((nt_group[used] == c["mbon_group"][used]).sum())
    lines.append(f"NT cross-check (glutamate=avoid): {agree}/{int(used.sum())} agree")
    have_pos = sum((c[f"{p}_pos"] != 0).any(axis=1).sum() for p in ("pn", "kc", "mbon", "dan"))
    lines.append(f"neurons with 3D skeleton position: {have_pos}/"
                 f"{sum(len(c[f'{p}_body']) for p in ('pn', 'kc', 'mbon', 'dan'))}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="audit an existing circuit.npz")
    args = ap.parse_args()
    if args.report:
        print(report(load()))
    else:
        print(report(extract()))


if __name__ == "__main__":
    main()
