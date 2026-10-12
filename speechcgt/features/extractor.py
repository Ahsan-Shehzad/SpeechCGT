from __future__ import annotations

from typing import Any, Optional

import torch

from ..constants import FEATURE_NODE_NAMES
from ..data.schemas import PreprocessedSample
from .acoustic import extract_acoustic_features
from .transcript import extract_transcript_features


def assemble_ordered_feature_vector(
    acoustic_features: dict[str, Optional[float]],
    transcript_features: dict[str, Optional[float]],
) -> tuple[torch.FloatTensor, torch.BoolTensor]:
    values = []
    mask = []
    merged = {**acoustic_features, **transcript_features}
    for feature_name in FEATURE_NODE_NAMES:
        value = merged.get(feature_name)
        if value is None:
            values.append(0.0)
            mask.append(False)
        else:
            values.append(float(value))
            mask.append(True)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


class SpeechFeatureExtractor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.audio_config = config["audio"]
        self.transcript_config = config["transcript"]

    def extract(self, sample: PreprocessedSample) -> tuple[torch.FloatTensor, torch.BoolTensor]:
        acoustic_features = extract_acoustic_features(sample, self.audio_config)
        transcript_features = extract_transcript_features(sample, self.transcript_config)
        return assemble_ordered_feature_vector(acoustic_features, transcript_features)

