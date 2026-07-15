import numpy as np
import pytest
from lumen.gating.dino_embed import embed

# DINOv2's CPU fallback emits harmless "xFormers is not available" UserWarnings
# during model construction; pytest surfaces them regardless of module filters.
pytestmark = pytest.mark.filterwarnings("ignore:xFormers is not available")


def test_embed_returns_384d_finite_vector():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(200, 260, 3), dtype=np.uint8)
    feat = embed(img)
    assert feat.shape == (384,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_embed_is_deterministic():
    img = np.full((128, 128, 3), 127, dtype=np.uint8)
    a = embed(img)
    b = embed(img)
    assert np.allclose(a, b, atol=1e-5)
