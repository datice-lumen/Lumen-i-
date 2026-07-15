"""Frozen DINOv2-S embedder (same backbone/transform as the classifier)."""
from __future__ import annotations

import functools
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_RESIZE = 448  # matches train.py

_transform = transforms.Compose([
    transforms.Resize((_RESIZE, _RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


@functools.lru_cache(maxsize=1)
def _load_model():
    # DINOv2 emits harmless "xFormers is not available" warnings when it falls
    # back to plain attention (no xFormers on this CPU-only box). Suppress them
    # locally at model-construction time rather than mutating the global filter.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is not available.*")
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False)
    model.eval()
    return model


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr.astype(np.uint8), "RGB")
    raise ValueError(f"unsupported image shape {getattr(arr, 'shape', None)}")


def embed(image) -> np.ndarray:
    """Return the 384-d DINOv2-S CLS embedding of an RGB image."""
    model = _load_model()
    tensor = _transform(_to_pil(image)).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)
