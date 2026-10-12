from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data

from .schema import GraphSchema


class NeurolinguisticGraphBuilder:
    def __init__(self, graph_schema: GraphSchema) -> None:
        self.graph_schema = graph_schema

    def build_subject_graph(
        self,
        x_hat: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
        label: Optional[int],
        graph_schema: GraphSchema | None = None,
    ) -> Data:
        return build_subject_graph(x_hat, feature_mask, label, graph_schema or self.graph_schema)


def build_subject_graph(
    x_hat: torch.FloatTensor,
    feature_mask: torch.BoolTensor,
    label: Optional[int],
    graph_schema: GraphSchema,
) -> Data:
    node_value = torch.zeros(len(graph_schema.node_names), 1, dtype=torch.float32)
    node_value[: len(x_hat), 0] = x_hat.float()
    data = Data(
        node_value=node_value,
        node_type=graph_schema.node_type_ids.clone(),
        level_id=graph_schema.level_ids.clone(),
        feature_mask=feature_mask.clone(),
        edge_index=graph_schema.edge_index.clone(),
        edge_type=graph_schema.edge_type.clone(),
    )
    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float32)
    return data

