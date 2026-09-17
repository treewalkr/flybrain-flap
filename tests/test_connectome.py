"""Tests for the MaleCNS circuit extraction (real data if present, synthetic otherwise)."""

from __future__ import annotations

import numpy as np
import pytest

from flybrain_flap.connectome import (APPROACH, AVOID, EXCLUDED,
                                      parse_compartments)


def test_parse_compartments():
    assert parse_compartments("MBON01(y5B'2a)_R") == ["g5", "b'2"]
    assert parse_compartments("MBON11(a'2a)>something") == ["a'2"]
    assert parse_compartments("DAN-f1<vGlutABN(y1ped>c)") == ["g1"]
    assert parse_compartments("no parentheses here") == []


@pytest.mark.skipif(not __import__("pathlib").Path("data/circuit.npz").exists(),
                    reason="run python -m flybrain_flap.fetch_connectome first")
def test_extracted_circuit_facts():
    from flybrain_flap.connectome import load
    c = load()
    # published MaleCNS facts
    assert len(c["pn_body"]) == 686
    assert 3800 <= (c["pn_kc"] > 0).sum(axis=0).clip(min=0).astype(bool).sum() <= 4100
    kc_in = (c["pn_kc"] > 0).sum(axis=0)
    kc_in = kc_in[kc_in > 0]
    assert 5.0 < kc_in.mean() < 7.0          # ~5.9 PNs per KC
    assert (c["pn_kc"] > 0).sum() == 22586   # PN->KC edges
    assert (c["kc_mbon"] > 0).sum() == 61210  # KC->MBON edges
    assert len(c["mbon_body"]) == 97
    assert len(c["dan_body"]) == 332
    # MBON groups partition sensibly; compartments have exactly one class
    assert ((c["mbon_group"] == APPROACH) | (c["mbon_group"] == AVOID)
            | (c["mbon_group"] == EXCLUDED)).all()
    assert set(c["comp_class"]) <= {APPROACH, AVOID, EXCLUDED} | {0, 1}
    # every neuron has a 3D position
    for pop in ("pn", "kc", "mbon", "dan"):
        pos = c[f"{pop}_pos"]
        assert pos.shape == (len(c[f"{pop}_body"]), 3)
        assert np.isfinite(pos).all()
    assert (c["kc_pos"] != 0).any()
