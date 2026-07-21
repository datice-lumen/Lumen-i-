import numpy as np
from PIL import Image
from lumen.gating.dino_embed import embed


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


def test_embed_accepts_pil_image_matching_numpy():
    # The public interface must also accept a PIL image (later tasks rely on it).
    arr = np.full((128, 128, 3), 127, dtype=np.uint8)
    from_numpy = embed(arr)
    from_pil = embed(Image.fromarray(arr, "RGB"))
    assert from_pil.shape == (384,)
    assert from_pil.dtype == np.float32
    assert np.allclose(from_numpy, from_pil, atol=1e-5)
