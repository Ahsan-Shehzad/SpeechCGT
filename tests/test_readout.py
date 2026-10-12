import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from torch_geometric.data import Batch

from speechcgt.graph.builder import build_subject_graph
from speechcgt.models.framework import SpeechCGTFramework


def test_readout_row_normalization(toy_config, graph_schema):
    model = SpeechCGTFramework(toy_config, graph_schema)
    x_hat = torch.randn(1, 24)
    mask = torch.ones(1, 24, dtype=torch.bool)
    batch = Batch.from_data_list([build_subject_graph(x_hat[0], mask[0], 1, graph_schema)])
    output = model(batch, x_hat, mask)
    assert torch.allclose(output.path_fc.sum(dim=-1), torch.ones_like(output.path_fc.sum(dim=-1)), atol=1e-4)
    assert torch.allclose(output.path_cd.sum(dim=-1), torch.ones_like(output.path_cd.sum(dim=-1)), atol=1e-4)
