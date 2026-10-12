from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_classification_loss(
    ci_logit: torch.FloatTensor,
    labels: torch.FloatTensor,
    pos_weight: torch.FloatTensor,
) -> torch.FloatTensor:
    return F.binary_cross_entropy_with_logits(ci_logit, labels, pos_weight=pos_weight)


def compute_consistency_loss(
    u_function: torch.FloatTensor,
    propagated_function: torch.FloatTensor,
    u_domain: torch.FloatTensor,
    propagated_domain: torch.FloatTensor,
) -> torch.FloatTensor:
    function_term = torch.mean((u_function - propagated_function) ** 2)
    domain_term = torch.mean((u_domain - propagated_domain) ** 2)
    return function_term + domain_term


def compute_gate_sparsity_loss(
    gate_fc: torch.FloatTensor,
    gate_cd: torch.FloatTensor,
) -> torch.FloatTensor:
    return gate_fc.abs().mean() + gate_cd.abs().mean()


def compute_total_loss(
    cls_loss: torch.FloatTensor,
    consistency_loss: torch.FloatTensor,
    gate_loss: torch.FloatTensor,
    consistency_weight: float,
    gate_sparsity_weight: float,
) -> dict[str, torch.FloatTensor]:
    total = cls_loss + consistency_weight * consistency_loss + gate_sparsity_weight * gate_loss
    return {
        "total": total,
        "classification": cls_loss,
        "consistency": consistency_loss,
        "gate_sparsity": gate_loss,
    }

