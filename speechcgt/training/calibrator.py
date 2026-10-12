from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class TemperatureScaler:
    def __init__(self, config: dict[str, Any]) -> None:
        self.max_iter = int(config["calibration"].get("max_iter", 200))
        self.temperature = 1.0

    def fit(self, logits: torch.FloatTensor, labels: torch.FloatTensor) -> float:
        logits = logits.detach().float()
        labels = labels.detach().float()
        log_temperature = torch.nn.Parameter(torch.zeros(1, device=logits.device))
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            temperature = torch.exp(log_temperature).clamp_min(1e-6)
            loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temperature.detach()).cpu().item())
        return self.temperature

    def transform(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        temperature = max(self.temperature, 1e-6)
        return torch.sigmoid(logits / temperature)

