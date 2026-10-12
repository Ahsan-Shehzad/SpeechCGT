from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..constants import FEATURE_NODE_NAMES
from ..utils.io import save_json

PROSODIC_FEATURES = {"pitch_variability", "pitch_range", "loudness_variability", "prosodic_reset"}
PHONATORY_FEATURES = {"jitter_shimmer", "harmonics_to_noise", "cepstral_prominence", "formant_stability"}
TEMPORAL_FEATURES = {
    "speech_rate",
    "articulation_rate",
    "silence_ratio",
    "mean_pause_duration",
    "pause_frequency",
    "filled_pause_rate",
    "self_repair_rate",
    "utterance_length",
}
LEXICAL_FEATURES = {"lexical_richness", "content_word_density", "pronoun_dependency"}
SYNTACTIC_FEATURES = {"syntactic_depth", "clause_linking_ratio"}


@dataclass
class RankedPathway:
    rank: int
    source_name: str
    target_name: str
    source_index: int
    target_index: int
    relation_name: str
    weight: float


@dataclass
class InferenceReport:
    sample_id: str
    participant_id: str
    ci_logit: float
    ci_prob_raw: float
    ci_prob_calibrated: float
    predicted_label: int
    feature_scores: list[float]
    function_scores: list[float]
    domain_scores: list[float]
    top_feature_to_function: list[RankedPathway]
    top_function_to_domain: list[RankedPathway]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["top_feature_to_function"] = [asdict(item) for item in self.top_feature_to_function]
        data["top_function_to_domain"] = [asdict(item) for item in self.top_function_to_domain]
        return data


def _fc_relation_name(feature_name: str) -> str:
    if feature_name in PROSODIC_FEATURES:
        return "fc_prosodic"
    if feature_name in PHONATORY_FEATURES:
        return "fc_phonatory_articulatory"
    if feature_name in TEMPORAL_FEATURES:
        return "fc_temporal_fluency"
    if feature_name in LEXICAL_FEATURES:
        return "fc_lexical"
    if feature_name in SYNTACTIC_FEATURES:
        return "fc_syntactic"
    return "fc_discourse"


def rank_feature_to_function_paths(
    path_fc: torch.FloatTensor,
    top_k: int,
    feature_names: list[str],
    function_names: list[str],
) -> list[RankedPathway]:
    candidates: list[RankedPathway] = []
    matrix = path_fc.detach().cpu()
    for feature_index, feature_name in enumerate(feature_names):
        for function_index, function_name in enumerate(function_names):
            weight = float(matrix[feature_index, function_index].item())
            if weight <= 0:
                continue
            candidates.append(
                RankedPathway(
                    rank=0,
                    source_name=feature_name,
                    target_name=function_name,
                    source_index=feature_index,
                    target_index=function_index,
                    relation_name=_fc_relation_name(feature_name),
                    weight=weight,
                )
            )
    candidates.sort(key=lambda item: item.weight, reverse=True)
    for rank, item in enumerate(candidates[:top_k], start=1):
        item.rank = rank
    return candidates[:top_k]


def rank_function_to_domain_paths(
    path_cd: torch.FloatTensor,
    top_k: int,
    function_names: list[str],
    domain_names: list[str],
) -> list[RankedPathway]:
    candidates: list[RankedPathway] = []
    matrix = path_cd.detach().cpu()
    for function_index, function_name in enumerate(function_names):
        for domain_index, domain_name in enumerate(domain_names):
            weight = float(matrix[function_index, domain_index].item())
            if weight <= 0:
                continue
            candidates.append(
                RankedPathway(
                    rank=0,
                    source_name=function_name,
                    target_name=domain_name,
                    source_index=function_index,
                    target_index=domain_index,
                    relation_name="cd_hierarchy",
                    weight=weight,
                )
            )
    candidates.sort(key=lambda item: item.weight, reverse=True)
    for rank, item in enumerate(candidates[:top_k], start=1):
        item.rank = rank
    return candidates[:top_k]


def export_cognitive_map_json(report: InferenceReport, output_path: str | Path) -> None:
    save_json(report.to_dict(), output_path)

