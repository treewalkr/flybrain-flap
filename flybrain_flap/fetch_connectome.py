"""Fetch the MaleCNS v1.0 connectome data the real mushroom body is built from.

    .venv/bin/python -m flybrain_flap.fetch_connectome            # everything
    .venv/bin/python -m flybrain_flap.fetch_connectome --report   # just counts

Downloads into data/ (gitignored):
    data/malecns/   three flat-connectome feather tables (Janelia FlyEM)
    data/ply/       JRCFIB2022M brain shell mesh
    data/swc/       per-neuron SWC skeletons of the mushroom-body populations

Every primary file is sha256-pinned; a present file with the right digest is
not downloaded again. Skeletons download concurrently and are parsed into
packed arrays under data/npz/<population>.npz (points + line topology).

Data sources (CC BY 4.0):
    MaleCNS v1.0 flat connectome, Janelia FlyEM
    JRCFIB2022M brain shell, FlyEM MaleCNS ROI meshes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path("data")
MALECNS = DATA / "malecns"
SWC_DIR = DATA / "swc"
NPZ_DIR = DATA / "npz"
PLY_DIR = DATA / "ply"

BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0"
FLAT = f"{BASE}/connectome-data/flat-connectome/"
SWC_URL = f"{BASE}/segmentation/skeletons-malecns/skeletons-swc/{{body_id}}.swc"
MESH_URL = "https://storage.googleapis.com/flyem-male-cns/rois/pointcloud-shells/"

# population class labels in the annotations table
CLASSES = {"pn": "ALPN", "kc": "Kenyon_Cell", "mbon": "MBON", "dan": "DAN"}

FILES = {
    MALECNS / "body-annotations-male-cns-v1.0-minconf-0.5.feather": (
        FLAT + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        14483314, "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
    MALECNS / "body-neurotransmitters-male-cns-v1.0.feather": (
        FLAT + "body-neurotransmitters-male-cns-v1.0.feather",
        43282834, "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
    MALECNS / "connectome-weights-male-cns-v1.0-minconf-0.5.feather": (
        FLAT + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        1051241946, "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
    PLY_DIR / "JRCFIB2022M_brain.ply": (
        MESH_URL + "JRCFIB2022M_brain.ply",
        1255587, "13ea41a7ce2677eae00738c23b2be3c2f29ac17c6976c2209998a85efccaa427"),
}

VOXEL_NM = 8.0  # SWC coordinates are in 8 nm voxel units


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path, size: int, sha: str) -> None:
    if path.exists() and path.stat().st_size == size and _sha(path) == sha:
        print(f"  ok (cached): {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    print(f"  download:   {path.name} ({size >> 20} MB)")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"{path}: {tmp.stat().st_size} bytes, expected {size}")
    if _sha(tmp) != sha:
        tmp.unlink()
        raise RuntimeError(f"{path}: sha256 mismatch")
    tmp.rename(path)


def body_ids() -> dict[str, np.ndarray]:
    """Body IDs of the traced mushroom-body populations, from the annotations."""
    import pandas as pd

    ann = pd.read_feather(MALECNS / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ann = ann[ann["status"] == "Traced"]
    ids = {}
    for pop, cls in CLASSES.items():
        frame = ann[ann["class"] == cls]
        types = frame["type"].fillna("").astype(str)
        if pop == "dan":
            frame = frame[types.str.startswith(("PAM", "PPL1"))]
        ids[pop] = np.sort(frame["bodyId"].to_numpy(np.int64))
    return ids


def parse_swc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """SWC -> (points um (N,3), edges (M,2) index pairs). Drops isolated samples."""
    pts, parent = [], []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            # sample, structure, x, y, z, radius, parent
            pts.append((float(parts[2]), float(parts[3]), float(parts[4])))
            parent.append(int(parts[6]))
    pts_arr = np.array(pts, dtype=np.float32) * (VOXEL_NM / 1000.0)
    parent_arr = np.array(parent, dtype=np.int64)
    edges = np.stack([np.arange(1, len(parent_arr) + 1, dtype=np.int64), parent_arr], axis=1)
    edges = edges[edges[:, 1] > 0] - 1  # to 0-based
    return pts_arr, edges


def fetch_skeletons(pop: str, ids: np.ndarray, workers: int = 16) -> Path:
    """Download + parse the SWC skeleton of every neuron in one population."""
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    SWC_DIR.mkdir(parents=True, exist_ok=True)
    out = NPZ_DIR / f"{pop}.npz"
    done = set()
    if out.exists():
        with np.load(out) as d:
            done = set(d["body_id"].tolist())
        if done == set(ids.tolist()):
            print(f"  ok (cached): {out.name}")
            return out

    def one(body: int):
        swc = SWC_DIR / f"{body}.swc"
        if not swc.exists():
            urllib.request.urlretrieve(SWC_URL.format(body_id=body), swc)
        return body, *parse_swc(swc)

    todo = [int(b) for b in ids if int(b) not in done]
    print(f"  skeletons:  {pop}: {len(todo)} to download ({len(done)} cached)")
    entries = {}
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for body, pts, edges in ex.map(one, todo):
                entries[body] = (pts, edges)
    # merge with previous cache
    if done:
        with np.load(out) as d:
            for i, body in enumerate(d["body_id"]):
                entries[int(body)] = (d[f"pts_{i}"], d[f"edges_{i}"])
    bodies = sorted(entries)
    arrays = {"body_id": np.array(bodies, dtype=np.int64)}
    for i, b in enumerate(bodies):
        pts, edges = entries[b]
        arrays[f"pts_{i}"] = pts
        arrays[f"edges_{i}"] = edges
    np.savez_compressed(out, **arrays)
    return out


def skeleton_report() -> dict:
    """Per-population summary used by both --report and tests."""
    import pandas as pd

    ids = body_ids()
    report = {"counts": {k: int(len(v)) for k, v in ids.items()}}
    w = pd.read_feather(MALECNS / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    idx = {p: {int(b): i for i, b in enumerate(ids[p])} for p in ids}
    for pre, post in (("pn", "kc"), ("kc", "mbon"), ("dan", "mbon")):
        sub = w[w["body_pre"].isin(idx[pre]) & w["body_post"].isin(idx[post])]
        report[f"{pre}->{post}"] = {
            "edges": int(len(sub)), "synapses": int(sub["weight"].sum())
        }
    if (NPZ_DIR / "kc.npz").exists():
        with np.load(NPZ_DIR / "kc.npz") as d:
            n = len(d["body_id"])
            lengths = [len(d[f"pts_{i}"]) for i in range(min(n, 50))]
            report["skeletons"] = {"kc_cached": n,
                                   "mean_samples": float(np.mean(lengths))}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-skeletons", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report", action="store_true",
                    help="print the population report (requires data present)")
    args = ap.parse_args()

    if args.report:
        print(json.dumps(skeleton_report(), indent=2))
        return

    print("primary files:")
    for path, (url, size, sha) in FILES.items():
        download(url, path, size, sha)

    ids = body_ids()
    print("populations:", {k: len(v) for k, v in ids.items()})
    (DATA / "body_ids.json").write_text(
        json.dumps({k: v.tolist() for k, v in ids.items()}))

    if not args.skip_skeletons:
        print("skeletons:")
        for pop in CLASSES:
            fetch_skeletons(pop, ids[pop], workers=args.workers)

    print(json.dumps(skeleton_report(), indent=2))


if __name__ == "__main__":
    main()
