"""Frozen DINOv2-S embedder (same backbone/transform as the classifier)."""
from __future__ import annotations

import functools
import warnings

import numpy as np
import torch
from PIL import Image

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_RESIZE = 448  # matches train.py

_MEAN = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(_IMAGENET_STD).view(3, 1, 1)


def _transform(pil: Image.Image) -> torch.Tensor:
    """Resize(448, bilinear) + ToTensor + ImageNet-normalize.

    Implemented with PIL+torch (no torchvision) so the web-app environment, which
    ships torch but not torchvision, can import this. Verified bit-identical to the
    torchvision Compose it replaces, so the fitted gate stays valid.
    """
    pil = pil.resize((_RESIZE, _RESIZE), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0   # HxWx3 in [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)   # CxHxW
    return (tensor - _MEAN) / _STD


_SHARED_MODEL = None


def set_backbone(model) -> None:
    """Inject an already-loaded frozen DINOv2-S so `embed` reuses it.

    The web app loads one backbone at startup for the classifier; without this the
    gate would `torch.hub.load` a second, independent copy on the first request,
    which on a small (1-2 GB) container is enough to get the process OOM-killed.
    """
    global _SHARED_MODEL
    _SHARED_MODEL = model


@functools.lru_cache(maxsize=1)
def _load_model():
    if _SHARED_MODEL is not None:
        return _SHARED_MODEL
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
