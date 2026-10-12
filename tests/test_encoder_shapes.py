import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
from torch_geometric.data import Batch

from speechcgt.graph.builder import build_subject_graph
from speechcgt.models.framework import SpeechCGTFramework


def test_model_shape_contracts(toy_config, graph_schema):
    model = SpeechCGTFramework(toy_config, graph_schema)
    x_hat = torch.randn(2, 24)
    mask = torch.ones(2, 24, dtype=torch.bool)
    graphs = [build_subject_graph(x_hat[i], mask[i], int(i % 2), graph_schema) for i in range(2)]
    batch = Batch.from_data_list(graphs)
    output, aux = model(batch, x_hat, mask, return_aux=True)
    assert tuple(output.feature_scores.shape) == (2, 24)
    assert tuple(output.function_scores.shape) == (2, 10)
    assert tuple(output.domain_scores.shape) == (2, 4)
    assert tuple(output.path_fc.shape) == (2, 24, 10)
    assert tuple(output.path_cd.shape) == (2, 10, 4)
    assert tuple(aux["attention_all"].shape) == (2, 8, 111)
    assert tuple(aux["gate_all"].shape) == (2, 8, 111)
