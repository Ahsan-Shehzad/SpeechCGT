import pytest

pytest.importorskip("torch")

import torch

from speechcgt.graph.builder import build_subject_graph


def test_graph_builder_shapes(graph_schema):
    x_hat = torch.zeros(24, dtype=torch.float32)
    mask = torch.ones(24, dtype=torch.bool)
    graph = build_subject_graph(x_hat, mask, 1, graph_schema)
    assert tuple(graph.edge_index.shape) == (2, 111)
    assert graph_schema.fc_edge_mask.sum().item() == 59
    assert graph_schema.cd_edge_mask.sum().item() == 14
    assert graph_schema.self_loop_mask.sum().item() == 38
    assert tuple(graph.node_value.shape) == (38, 1)
    assert tuple(graph.feature_mask.shape) == (24,)
