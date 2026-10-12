from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..graph.schema import GraphSchema


class CognitiveReadoutHead(nn.Module):
    def __init__(self, config: dict[str, Any], graph_schema: GraphSchema) -> None:
        super().__init__()
        hidden_dim = int(config["model"]["hidden_dim"])
        self.graph_schema = graph_schema
        self.feature_scorer = nn.Linear(hidden_dim, 1)
        self.function_scorer = nn.Linear(hidden_dim, 1)
        self.domain_scorer = nn.Linear(hidden_dim, 1)
        for layer in [self.feature_scorer, self.function_scorer, self.domain_scorer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        final_attention: torch.FloatTensor,
        final_gates: torch.FloatTensor,
        graph_schema: GraphSchema,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        device = hidden_states.device
        feature_states = hidden_states[:, graph_schema.feature_node_indices.to(device), :]
        function_states = hidden_states[:, graph_schema.function_node_indices.to(device), :]
        domain_states = hidden_states[:, graph_schema.domain_node_indices.to(device), :]
        u_feature = torch.nn.functional.softplus(self.feature_scorer(feature_states).squeeze(-1))
        u_function = torch.nn.functional.softplus(self.function_scorer(function_states).squeeze(-1))
        u_domain = torch.nn.functional.softplus(self.domain_scorer(domain_states).squeeze(-1))
        gated_attention = (final_attention * final_gates).mean(dim=1)
        path_fc = torch.zeros(hidden_states.size(0), 24, 10, device=device)
        path_cd = torch.zeros(hidden_states.size(0), 10, 4, device=device)
        fc_edges = graph_schema.edge_index[:, graph_schema.fc_edge_mask].to(device)
        cd_edges = graph_schema.edge_index[:, graph_schema.cd_edge_mask].to(device)
        fc_weights = gated_attention[:, graph_schema.fc_edge_mask.to(device)]
        cd_weights = gated_attention[:, graph_schema.cd_edge_mask.to(device)]
        for edge_position in range(fc_edges.size(1)):
            source = int(fc_edges[0, edge_position].item())
            target = int(fc_edges[1, edge_position].item() - 24)
            path_fc[:, source, target] = fc_weights[:, edge_position]
        for edge_position in range(cd_edges.size(1)):
            source = int(cd_edges[0, edge_position].item() - 24)
            target = int(cd_edges[1, edge_position].item() - 34)
            path_cd[:, source, target] = cd_weights[:, edge_position]
        path_fc = path_fc / path_fc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        path_cd = path_cd / path_cd.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        propagated_function = torch.bmm(path_fc.transpose(1, 2), u_feature.unsqueeze(-1)).squeeze(-1)
        function_scores = 0.5 * (u_function + propagated_function)
        propagated_domain = torch.bmm(path_cd.transpose(1, 2), function_scores.unsqueeze(-1)).squeeze(-1)
        domain_scores = 0.5 * (u_domain + propagated_domain)
        public_outputs = {
            "feature_scores": u_feature,
            "function_scores": function_scores,
            "domain_scores": domain_scores,
            "path_fc": path_fc,
            "path_cd": path_cd,
        }
        aux_outputs = {
            "u_feature": u_feature,
            "u_function": u_function,
            "u_domain": u_domain,
            "propagated_function": propagated_function,
            "propagated_domain": propagated_domain,
            "attention_all": final_attention,
            "gate_all": final_gates,
            "attention_fc": final_attention[:, :, graph_schema.fc_edge_mask.to(device)],
            "gate_fc": final_gates[:, :, graph_schema.fc_edge_mask.to(device)],
            "attention_cd": final_attention[:, :, graph_schema.cd_edge_mask.to(device)],
            "gate_cd": final_gates[:, :, graph_schema.cd_edge_mask.to(device)],
        }
        return public_outputs, aux_outputs

