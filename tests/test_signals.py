"""L4 — signal processing: recover composition & crystallinity from a DSC thermogram."""

from __future__ import annotations

import numpy as np
import pytest

from biopoly.signals import MELT_TEMP, extract_features, process_signal, synth_dsc

pytestmark = pytest.mark.layer(4)  # features, splitting & signal DSP


def _features_for(pfrac, **kw):
    x, y = synth_dsc(pfrac, **kw)
    return extract_features(x, process_signal(x, y))


def test_synth_dsc_shape_and_finite():
    x, y = synth_dsc({"PLA": 1.0})
    assert x.shape == y.shape and x.ndim == 1
    assert np.all(np.isfinite(y))


def test_single_polymer_peak_near_its_melt_temp():
    feats = _features_for({"PLA": 1.0})
    assert feats["n_peaks"] >= 1
    assert abs(feats["dominant_temp_c"] - MELT_TEMP["PLA"]) < 8.0


def test_features_discriminate_composition():
    # PLA melts hot (~155 C), PCL cold (~60 C): the dominant peak must separate them.
    pla = _features_for({"PLA": 1.0})
    pcl = _features_for({"PCL": 1.0})
    assert pla["dominant_temp_c"] - pcl["dominant_temp_c"] > 60.0


def test_two_well_separated_polymers_give_two_peaks():
    feats = _features_for({"PLA": 0.5, "PCL": 0.5})
    assert feats["n_peaks"] >= 2


def test_smoothing_reduces_noise():
    rng = np.random.default_rng(0)
    x, y = synth_dsc({"PLA": 1.0}, rng=rng)
    proc = process_signal(x, y)
    # high-frequency roughness (variance of the first difference) should drop
    assert np.var(np.diff(proc)) < np.var(np.diff(y))


def test_nucleating_agent_sharpens_peaks():
    # Nucleation raises crystallinity → a narrower melt endotherm (mechanism check).
    plain = _features_for({"PBS": 1.0})
    nucleated = _features_for({"PBS": 1.0}, nucleating=0.3)
    assert nucleated["mean_width_c"] < plain["mean_width_c"]


def test_extract_features_deterministic():
    x, y = synth_dsc({"PLA": 0.6, "PBS": 0.4})  # no rng → deterministic
    assert extract_features(x, process_signal(x, y)) == extract_features(x, process_signal(x, y))


def test_no_signal_returns_zeroed_features():
    x, y = synth_dsc({})  # nothing melts → flat after baseline correction
    feats = extract_features(x, process_signal(x, y))
    assert feats["n_peaks"] == 0
    assert feats["total_area"] == 0.0
