"""One-class Mahalanobis OOD gate over frozen embeddings.

Fit = summary statistics only (mean, shrunk covariance, threshold). No training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class OODGate:
    mean: np.ndarray       # (D,)
    precision: np.ndarray  # (D, D) inverse covariance
    threshold: float       # Mahalanobis distance cutoff

    @classmethod
    def fit(cls, features: np.ndarray, percentile: float = 99.0) -> "OODGate":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"expected 2D (N, D), got shape {features.shape}")
        mean = features.mean(axis=0)
        precision = LedoitWolf().fit(features).precision_
        distances = cls._distances(features, mean, precision)
        threshold = float(np.percentile(distances, percentile))
        return cls(mean=mean, precision=precision, threshold=threshold)

    @staticmethod
    def _distances(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        centered = x - mean
        quad = np.einsum("ij,jk,ik->i", centered, precision, centered)
        return np.sqrt(np.clip(quad, 0.0, None))

    def score(self, feature: np.ndarray) -> float:
        feature = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(feature)):
            raise ValueError("feature contains NaN/inf")
        return float(self._distances(feature, self.mean, self.precision)[0])

    def passes(self, feature: np.ndarray) -> bool:
        return self.score(feature) <= self.threshold

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, precision=self.precision,
                 threshold=np.array(self.threshold, dtype=np.float64))

    @classmethod
    def load(cls, path: str) -> "OODGate":
        data = np.load(path)
        return cls(mean=data["mean"], precision=data["precision"],
                   threshold=float(data["threshold"]))
