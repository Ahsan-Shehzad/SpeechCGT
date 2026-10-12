from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    probabilities: torch.FloatTensor,
    labels: torch.FloatTensor,
    threshold: float,
) -> dict[str, float]:
    probs = probabilities.detach().cpu().numpy().reshape(-1)
    y_true = labels.detach().cpu().numpy().reshape(-1)
    y_pred = (probs >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

