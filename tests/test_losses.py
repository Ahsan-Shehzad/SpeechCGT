import pytest

pytest.importorskip("torch")

import torch

from speechcgt.training.losses import (
    compute_classification_loss,
    compute_consistency_loss,
    compute_gate_sparsity_loss,
    compute_total_loss,
)


def test_loss_composition():
    logits = torch.tensor([[0.5], [-0.5]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    cls = compute_classification_loss(logits, labels, torch.tensor([1.0]))
    cons = compute_consistency_loss(torch.ones(2, 10), torch.zeros(2, 10), torch.ones(2, 4), torch.zeros(2, 4))
    gate = compute_gate_sparsity_loss(torch.ones(2, 8, 59), torch.ones(2, 8, 14))
    total = compute_total_loss(cls, cons, gate, 0.2, 1e-4)
    assert set(total.keys()) == {"total", "classification", "consistency", "gate_sparsity"}
    assert total["total"].item() > cls.item()
