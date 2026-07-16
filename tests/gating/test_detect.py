import numpy as np
import pytest

from lumen.gating import detect
from lumen.gating.detect import detect_lesion


class _Gate:
    """Minimal fitted-gate stand-in: fixed score + threshold."""
    def __init__(self, score, threshold):
        self._s = score
        self.threshold = threshold

    def score(self, feature):
        return self._s


@pytest.fixture
def stub_embed(monkeypatch):
    # Detection routing is tested hermetically — never load DINOv2.
    monkeypatch.setattr(detect, "embed", lambda img: np.zeros(384, dtype=np.float32))


def _img():
    return np.full((64, 64, 3), 127, dtype=np.uint8)


def test_unreadable_path_returns_error(stub_embed):
    out = detect_lesion("/no/such/file.jpg", _Gate(0.0, 1.0))
    assert out["status"] == "error"
    assert out["is_skin"] is False
    assert out["reason"] == "unreadable"
    assert out["score"] is None


def test_in_distribution_is_skin(stub_embed):
    out = detect_lesion(_img(), _Gate(0.5, 1.0))
    assert out["status"] == "skin"
    assert out["is_skin"] is True
    assert out["score"] == 0.5
    assert out["reason"] == "in_distribution"


def test_far_point_is_unclassified(stub_embed):
    out = detect_lesion(_img(), _Gate(5.0, 1.0))
    assert out["status"] == "unclassified"
    assert out["is_skin"] is False
    assert out["score"] == 5.0
    assert out["reason"] == "ood_distance>threshold"


def test_score_equal_to_threshold_is_skin(stub_embed):
    # Boundary: distance == threshold passes (<=), matching OODGate.passes.
    out = detect_lesion(_img(), _Gate(1.0, 1.0))
    assert out["status"] == "skin"
    assert out["is_skin"] is True
