from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from ..graph.schema import GraphSchema
from .attention import RelationalGatedMultiHeadAttention
from .embeddings import NodeEmbeddingLayer, PatientSummaryMLP


class PGCGTBlock(nn.Module):
    def __init__(self, config: dict[str, Any], graph_schema: GraphSchema) -> None:
        super().__init__()
        hidden_dim = int(config["model"]["hidden_dim"])
        ffn_dim = int(config["model"]["ffn_dim"])
        dropout = float(config["model"]["dropout"])
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = RelationalGatedMultiHeadAttention(config, graph_schema)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        patient_summary: torch.FloatTensor,
        graph_batch: Batch,
        graph_schema: GraphSchema,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        normalized = self.norm1(hidden_states)
        attention_out, attention_scores, gates = self.attention(normalized, patient_summary, graph_batch, graph_schema)
        hidden_states = hidden_states + self.dropout(attention_out)
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        return hidden_states, attention_scores, gates


class PGCGTEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], graph_schema: GraphSchema) -> None:
        super().__init__()
        self.graph_schema = graph_schema
        self.embedding_layer = NodeEmbeddingLayer(config, graph_schema)
        self.patient_summary = PatientSummaryMLP(config)
        self.blocks = nn.ModuleList(
            [PGCGTBlock(config, graph_schema) for _ in range(int(config["model"]["num_layers"]))]
        )

    def forward(
        self,
        graph_batch: Batch,
        x_hat: torch.FloatTensor,
        feature_mask: torch.BoolTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        hidden_states = self.embedding_layer(graph_batch, x_hat, feature_mask)
        patient_summary = self.patient_summary(x_hat, feature_mask.float())
        final_attention = None
        final_gates = None
        for block in self.blocks:
            hidden_states, final_attention, final_gates = block(
                hidden_states,
                patient_summary,
                graph_batch,
                self.graph_schema,
            )
        return hidden_states, final_attention, final_gates

