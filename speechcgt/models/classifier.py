from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MonotonicDomainClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(config["model"]["dropout"]))
        self.raw_weight = nn.Parameter(torch.zeros(4))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, domain_scores: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        domain_scores = self.dropout(domain_scores)
        weights = torch.nn.functional.softplus(self.raw_weight)
        ci_logit = torch.sum(domain_scores * weights.unsqueeze(0), dim=-1, keepdim=True) + self.bias
        return ci_logit, torch.sigmoid(ci_logit)

