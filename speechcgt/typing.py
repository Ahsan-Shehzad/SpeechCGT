from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias

import torch

FeatureTensor: TypeAlias = torch.FloatTensor
FeatureMask: TypeAlias = torch.BoolTensor


@dataclass
class SpeechCGTOutput:
    ci_logit: torch.FloatTensor
    ci_prob: torch.FloatTensor
    feature_scores: torch.FloatTensor
    function_scores: torch.FloatTensor
    domain_scores: torch.FloatTensor
    path_fc: torch.FloatTensor
    path_cd: torch.FloatTensor


TensorDict: TypeAlias = dict[str, torch.Tensor]
OptionalTensor: TypeAlias = Optional[torch.Tensor]

