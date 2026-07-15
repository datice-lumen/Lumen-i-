import numpy as np
import pytest
from lumen.gating.ood_gate import OODGate


def _cluster(n=2000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, d))


def test_in_distribution_point_passes():
    gate = OODGate.fit(_cluster(), percentile=99.0)
    assert gate.passes(np.zeros(16)) is True


def test_far_point_is_rejected():
    gate = OODGate.fit(_cluster(), percentile=99.0)
    assert gate.passes(np.full(16, 50.0)) is False


def test_threshold_rejects_about_one_percent_of_fit():
    feats = _cluster(n=5000, d=16)
    gate = OODGate.fit(feats, percentile=99.0)
    rejected = np.mean([not gate.passes(f) for f in feats])
    assert 0.0 <= rejected <= 0.03


def test_save_load_roundtrip(tmp_path):
    gate = OODGate.fit(_cluster(), percentile=99.0)
    p = tmp_path / "gate.npz"
    gate.save(str(p))
    loaded = OODGate.load(str(p))
    f = np.ones(16)
    assert loaded.threshold == pytest.approx(gate.threshold)
    assert loaded.score(f) == pytest.approx(gate.score(f), rel=1e-6)


def test_non_finite_feature_raises():
    gate = OODGate.fit(_cluster(), percentile=99.0)
    with pytest.raises(ValueError):
        gate.score(np.array([np.nan] * 16))


def test_fit_rejects_non_2d():
    with pytest.raises(ValueError):
        OODGate.fit(np.zeros(16))
