from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from ..graph.schema import GraphSchema


class NodeEmbeddingLayer(nn.Module):
    def __init__(self, config: dict[str, Any], graph_schema: GraphSchema) -> None:
        super().__init__()
        hidden_dim = int(config["model"]["hidden_dim"])
        self.hidden_dim = hidden_dim
        self.graph_schema = graph_schema
        self.feature_projection = nn.Linear(2, hidden_dim)
        self.node_id_embedding = nn.Embedding(len(graph_schema.node_names), hidden_dim)
        self.node_type_embedding = nn.Embedding(3, hidden_dim)
        self.level_embedding = nn.Embedding(3, hidden_dim)
        self.prototype_embeddings = nn.Parameter(torch.zeros(len(graph_schema.node_names), hidden_dim))
        nn.init.xavier_uniform_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.xavier_uniform_(self.prototype_embeddings)

    def forward(
        self,
        graph_batch: Batch,
        x_hat: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        del graph_batch
        batch_size = x_hat.size(0)
        device = x_hat.device
        feature_input = torch.stack([x_hat, feature_mask.float()], dim=-1)
        feature_embeddings = self.feature_projection(feature_input)
        node_ids = torch.arange(len(self.graph_schema.node_names), device=device)
        structural = (
            self.node_id_embedding(node_ids)
            + self.node_type_embedding(self.graph_schema.node_type_ids.to(device))
            + self.level_embedding(self.graph_schema.level_ids.to(device))
        )
        structural = structural.unsqueeze(0).expand(batch_size, -1, -1)
        hidden_states = structural + self.prototype_embeddings.unsqueeze(0)
        hidden_states[:, self.graph_schema.feature_node_indices.to(device), :] = (
            feature_embeddings + structural[:, self.graph_schema.feature_node_indices.to(device), :]
        )
        return hidden_states


class PatientSummaryMLP(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["model"]["patient_summary_dim"])
        self.network = nn.Sequential(
            nn.Linear(48, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x_hat: torch.FloatTensor,
        feature_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        features = torch.cat([x_hat, feature_mask.float()], dim=-1)
        return self.network(features)

