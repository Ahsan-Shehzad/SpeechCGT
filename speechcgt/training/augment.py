from __future__ import annotations

import random

import torch


def apply_modality_dropout(
    x_hat: torch.FloatTensor,
    p: float,
    acoustic_indices: list[int],
    transcript_indices: list[int],
) -> torch.FloatTensor:
    if p <= 0:
        return x_hat
    output = x_hat.clone()
    for batch_index in range(output.size(0)):
        if random.random() > p:
            continue
        drop_indices = acoustic_indices if random.random() < 0.5 else transcript_indices
        output[batch_index, drop_indices] = 0.0
    return output


def add_gaussian_feature_noise(
    x_hat: torch.FloatTensor,
    std: float,
) -> torch.FloatTensor:
    if std <= 0:
        return x_hat
    return x_hat + torch.randn_like(x_hat) * std

