"""Fused, metadata-aware model (model_10_6 / najbolji_10_6.pt).

Late-fusion architecture:
    frozen DINOv2-S + trainable TinyCNN -> VisionProj -> vision_emb (256)
    metadata (11-dim)                   -> MetaMLP    -> meta_emb   (16)
    concat(272) -> Classifier -> 1 logit

The sub-module definitions are copied verbatim from the training script
(model_10_6/train.py) so the saved state_dicts load unchanged. DINOv2-S is pulled
from torch.hub at load time (needs network on first run, then cached). This module
deliberately avoids torchvision so it imports cleanly for architecture/checkpoint
validation even where torchvision is absent.
"""

import numpy as np
import torch
import torch.nn as nn

# Dimensions — must match training / the checkpoint config.
DINO_DIM = 384
CNN_OUT_CHANNELS = 96
CNN_DIM = CNN_OUT_CHANNELS * 2            # avg + max pool concat = 192
VISION_DIM = DINO_DIM + CNN_DIM           # 576
VISION_EMB_DIM = 256
META_EMB_DIM = 16
META_DIM = 11                             # age_norm + age_missing + sex(3) + site(6)
FUSED_DIM = VISION_EMB_DIM + META_EMB_DIM  # 272

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TinyCNN(nn.Module):
    """Trainable conv feature extractor complementing frozen DINOv2 (192-dim)."""

    def __init__(self, out_channels=CNN_OUT_CHANNELS):
        super().__init__()
        # ReLUs are non-inplace (train.py used inplace=True): ReLU carries no
        # parameters, so this is identical for state_dict loading and output, and
        # it lets Grad-CAM register a backward hook on the final activation.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        feat = self.features(x)
        avg = self.avgpool(feat).flatten(1)
        mx = self.maxpool(feat).flatten(1)
        return self.dropout(torch.cat([avg, mx], dim=1))


class VisionProj(nn.Module):
    """Fuse DINO + TinyCNN (576) -> vision_emb (256). No BN on the output."""

    def __init__(self, in_dim=VISION_DIM, hidden=384, out_dim=VISION_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)


class MetaMLP(nn.Module):
    """Small MLP for metadata (age, sex, anatom_site). No BN on the output."""

    def __init__(self, in_dim=META_DIM, hidden=32, out_dim=META_EMB_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class Classifier(nn.Module):
    """Post-fusion classifier over concat(vision_emb, meta_emb) = 272."""

    def __init__(self, in_dim=FUSED_DIM, hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)


class FusedMetaModel(nn.Module):
    """Wraps the frozen DINO backbone + trainable heads. forward(img, meta) -> logits."""

    def __init__(self, dino, cnn, vision_proj, meta_mlp, classifier):
        super().__init__()
        self.dino = dino
        self.cnn = cnn
        self.vision_proj = vision_proj
        self.meta_mlp = meta_mlp
        self.classifier = classifier

    @property
    def gradcam_layer(self):
        """Last feature-map layer of the trainable CNN branch (96ch, 14x14)."""
        return self.cnn.features[-1]

    def forward(self, img, meta):
        with torch.no_grad():
            dino_feat = self.dino(img).float()
        cnn_feat = self.cnn(img)
        vision_emb = self.vision_proj(torch.cat([dino_feat, cnn_feat], dim=1))
        meta_emb = self.meta_mlp(meta)
        fused = torch.cat([vision_emb, meta_emb], dim=1)
        return self.classifier(fused)


def load_dino(device="cpu"):
    """Load frozen DINOv2-S (with registers) from torch.hub."""
    m = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False, trust_repo=True
    )
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m.to(device)


def load_fused_model(checkpoint_path, device="cpu"):
    """Build the fused model and load weights from the dict checkpoint.

    Returns (model, meta_cfg) where meta_cfg carries the metadata-encoder settings:
    {age_mean, age_std, sex_categories, site_categories, resize}.
    """
    # Trusted, repo-committed weights; only tensors + simple types -> weights_only.
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ck.get("config", {})

    dino = load_dino(device)
    cnn = TinyCNN(cfg.get("CNN_OUT_CHANNELS", CNN_OUT_CHANNELS)).to(device)
    vision_proj = VisionProj().to(device)
    meta_mlp = MetaMLP().to(device)
    classifier = Classifier().to(device)

    cnn.load_state_dict(ck["cnn"])
    vision_proj.load_state_dict(ck["vision_proj"])
    meta_mlp.load_state_dict(ck["meta_mlp"])
    classifier.load_state_dict(ck["classifier"])

    model = FusedMetaModel(dino, cnn, vision_proj, meta_mlp, classifier).to(device)
    model.eval()

    meta_cfg = {
        "age_mean": float(ck["age_mean"]),
        "age_std": float(ck["age_std"]),
        "sex_categories": list(cfg.get("SEX_CATEGORIES", ["male", "female", "unknown"])),
        "site_categories": list(
            cfg.get(
                "SITE_CATEGORIES",
                ["torso", "lower_extremity", "upper_extremity", "head_neck", "unknown", "palms_soles"],
            )
        ),
        "resize": int(cfg.get("RESIZE", 448)),
    }
    return model, meta_cfg


def load_submodules_for_validation(checkpoint_path, device="cpu"):
    """Load only the trainable heads (no DINO/torch.hub) to validate a checkpoint.

    Used by tests / environments without network. Returns the meta_cfg.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ck.get("config", {})
    cnn = TinyCNN(cfg.get("CNN_OUT_CHANNELS", CNN_OUT_CHANNELS))
    cnn.load_state_dict(ck["cnn"])
    VisionProj().load_state_dict(ck["vision_proj"])
    MetaMLP().load_state_dict(ck["meta_mlp"])
    Classifier().load_state_dict(ck["classifier"])
    return {
        "age_mean": float(ck["age_mean"]),
        "age_std": float(ck["age_std"]),
        "sex_categories": list(cfg.get("SEX_CATEGORIES", [])),
        "site_categories": list(cfg.get("SITE_CATEGORIES", [])),
        "resize": int(cfg.get("RESIZE", 448)),
    }


def image_to_tensor(img_rgb, device="cpu"):
    """Preprocessed RGB uint8 image -> normalized (1, 3, H, W) tensor (ImageNet stats)."""
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)


def encode_metadata(age, sex, anatom_site, meta_cfg, device="cpu"):
    """Encode one sample's metadata into the model's 11-dim vector.

    Mirrors model_10_6/train.py::encode_metadata for a single row. Unknown/blank
    values map to the 'unknown' one-hot bucket; missing age sets the age-missing flag
    and normalizes to 0. The category order follows meta_cfg exactly.

    Returns (tensor (1, META_DIM), resolved) where `resolved` echoes the values used.
    """
    sex_cats = meta_cfg["sex_categories"]
    site_cats = meta_cfg["site_categories"]

    age_val = None
    try:
        if age is not None and str(age).strip() != "":
            age_val = float(age)
    except (TypeError, ValueError):
        age_val = None

    if age_val is None:
        age_norm, age_missing = 0.0, 1.0
    else:
        age_norm = (age_val - meta_cfg["age_mean"]) / meta_cfg["age_std"]
        age_missing = 0.0

    sex_r = sex if sex in sex_cats else "unknown"
    site_r = anatom_site if anatom_site in site_cats else "unknown"

    sex_oh = [1.0 if c == sex_r else 0.0 for c in sex_cats]
    site_oh = [1.0 if c == site_r else 0.0 for c in site_cats]

    vec = [age_norm, age_missing] + sex_oh + site_oh
    tensor = torch.tensor([vec], dtype=torch.float32, device=device)
    resolved = {
        "age": None if age_val is None else (int(age_val) if float(age_val).is_integer() else age_val),
        "sex": sex_r,
        "anatom_site": site_r,
    }
    return tensor, resolved


def gradcam_fused(model, img_tensor, meta_tensor, out_size=448):
    """Grad-CAM over the TinyCNN branch of the fused model.

    Reflects the trainable CNN feature map only — DINO's contribution isn't captured.
    Returns a normalized [0, 1] heatmap of shape (out_size, out_size).
    """
    import cv2

    model.eval()
    target_layer = model.gradcam_layer
    activations, gradients = [], []

    def fwd_hook(_m, _inp, out):
        activations.append(out)

    def bwd_hook(_m, _gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img = img_tensor.clone().requires_grad_(True)
        logits = model(img, meta_tensor)
        model.zero_grad()
        logits[:, 0].backward()

        grads = gradients[0][0].detach().cpu().numpy()
        acts = activations[0][0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()

    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * acts, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cv2.resize(cam, (out_size, out_size))
