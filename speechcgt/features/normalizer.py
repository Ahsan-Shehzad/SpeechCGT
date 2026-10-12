from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..constants import FEATURE_NODE_NAMES


@dataclass
class FeatureStats:
    feature_names: list[str]
    mean: torch.FloatTensor
    std: torch.FloatTensor
    epsilon: float
    fitted_corpus: str
    fold_id: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mean"] = self.mean.tolist()
        payload["std"] = self.std.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureStats":
        return cls(
            feature_names=list(payload["feature_names"]),
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
            epsilon=float(payload["epsilon"]),
            fitted_corpus=str(payload["fitted_corpus"]),
            fold_id=int(payload["fold_id"]),
        )


def fit_feature_stats(
    feature_matrix: torch.FloatTensor,
    feature_mask: torch.BoolTensor,
    source_corpus: str,
    fold_id: int,
    epsilon: float = 1e-8,
) -> FeatureStats:
    observed = feature_mask.float()
    safe_counts = observed.sum(dim=0).clamp_min(1.0)
    mean = (feature_matrix * observed).sum(dim=0) / safe_counts
    centered = (feature_matrix - mean) * observed
    variance = (centered**2).sum(dim=0) / safe_counts
    std = torch.sqrt(variance).clamp_min(epsilon)
    return FeatureStats(
        feature_names=list(FEATURE_NODE_NAMES),
        mean=mean.detach().cpu(),
        std=std.detach().cpu(),
        epsilon=epsilon,
        fitted_corpus=source_corpus,
        fold_id=fold_id,
    )


def transform_features(
    x_raw: torch.FloatTensor,
    feature_mask: torch.BoolTensor,
    stats: FeatureStats,
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    mean = stats.mean.to(x_raw.device)
    std = stats.std.to(x_raw.device)
    mask_float = feature_mask.float()
    x_imputed = mask_float * x_raw + (1.0 - mask_float) * mean
    x_hat = (x_imputed - mean) / (std + stats.epsilon)
    return x_hat, feature_mask


class FeatureNormalizer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.epsilon = float(config["features"].get("normalization_epsilon", 1e-8))

    def fit_feature_stats(
        self,
        feature_matrix: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
        source_corpus: str,
        fold_id: int,
    ) -> FeatureStats:
        return fit_feature_stats(feature_matrix, feature_mask, source_corpus, fold_id, self.epsilon)

    def transform_features(
        self,
        x_raw: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
        stats: FeatureStats,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor]:
        return transform_features(x_raw, feature_mask, stats)
