from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch

from ..graph.schema import GraphSchema


class RelationalGatedMultiHeadAttention(nn.Module):
    def __init__(self, config: dict[str, Any], graph_schema: GraphSchema) -> None:
        super().__init__()
        hidden_dim = int(config["model"]["hidden_dim"])
        num_heads = int(config["model"]["num_heads"])
        head_dim = int(config["model"]["head_dim"])
        patient_summary_dim = int(config["model"]["patient_summary_dim"])
        if hidden_dim != num_heads * head_dim:
            raise ValueError("hidden_dim must equal num_heads * head_dim.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.graph_schema = graph_schema
        self.gate_lambda = float(config["model"]["gate_lambda"])
        self.gate_epsilon = float(config["model"]["gate_epsilon"])
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relation_bias = nn.Parameter(torch.zeros(len(graph_schema.relation_id_to_name), num_heads))
        self.relation_embedding = nn.Embedding(len(graph_schema.relation_id_to_name), hidden_dim)
        gate_input_dim = hidden_dim * 3 + patient_summary_dim
        self.gate_projection = nn.Linear(gate_input_dim, num_heads)
        self.dropout = nn.Dropout(float(config["model"]["dropout"]))
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.gate_projection]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        patient_summary: torch.FloatTensor,
        graph_batch: Batch,
        graph_schema: GraphSchema,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        del graph_batch
        batch_size, num_nodes, _ = hidden_states.shape
        device = hidden_states.device
        edge_index = graph_schema.edge_index.to(device)
        edge_type = graph_schema.edge_type.to(device)
        source_idx = edge_index[0]
        target_idx = edge_index[1]
        queries = self.q_proj(hidden_states).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        keys = self.k_proj(hidden_states).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        values = self.v_proj(hidden_states).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q_target = queries[:, target_idx, :, :]
        k_source = keys[:, source_idx, :, :]
        v_source = values[:, source_idx, :, :]
        base_scores = (q_target * k_source).sum(dim=-1) / math.sqrt(self.head_dim)
        relation_bias = self.relation_bias[edge_type].unsqueeze(0)
        relation_embedding = self.relation_embedding(edge_type).unsqueeze(0).expand(batch_size, -1, -1)
        gate_input = torch.cat(
            [
                hidden_states[:, target_idx, :],
                hidden_states[:, source_idx, :],
                patient_summary.unsqueeze(1).expand(-1, edge_index.size(1), -1),
                relation_embedding,
            ],
            dim=-1,
        )
        gates = torch.sigmoid(self.gate_projection(gate_input))
        scores = base_scores + relation_bias + self.gate_lambda * torch.log(gates + self.gate_epsilon)
        attention = torch.zeros_like(scores)
        for node_index in range(num_nodes):
            edge_mask = target_idx == node_index
            if torch.any(edge_mask):
                attention[:, edge_mask, :] = torch.softmax(scores[:, edge_mask, :], dim=1)
        aggregated = torch.zeros(batch_size, num_nodes, self.num_heads, self.head_dim, device=device)
        for edge_position in range(edge_index.size(1)):
            source = int(source_idx[edge_position].item())
            target = int(target_idx[edge_position].item())
            aggregated[:, target, :, :] += attention[:, edge_position, :].unsqueeze(-1) * v_source[:, edge_position, :, :]
        attended = aggregated.reshape(batch_size, num_nodes, self.hidden_dim)
        return self.dropout(self.out_proj(attended)), attention.permute(0, 2, 1), gates.permute(0, 2, 1)

