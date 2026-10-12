from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from ..constants import (
    DOMAIN_NODE_NAMES,
    FEATURE_NODE_NAMES,
    FUNCTION_NODE_NAMES,
    LEVEL_IDS,
    NODE_NAMES,
    NODE_TYPE_IDS,
    RELATION_IDS,
)
from .ontology import NeurolinguisticOntology


@dataclass
class GraphSchema:
    node_names: list[str]
    node_type_ids: torch.LongTensor
    level_ids: torch.LongTensor
    edge_index: torch.LongTensor
    edge_type: torch.LongTensor
    fc_edge_mask: torch.BoolTensor
    cd_edge_mask: torch.BoolTensor
    self_loop_mask: torch.BoolTensor
    relation_id_to_name: dict[int, str]
    feature_node_indices: torch.LongTensor
    function_node_indices: torch.LongTensor
    domain_node_indices: torch.LongTensor

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.tolist()
        return payload


class GraphSchemaBuilder:
    def __init__(self, ontology: NeurolinguisticOntology) -> None:
        self.ontology = ontology
        self.node_index = {name: idx for idx, name in enumerate(self.ontology.node_names)}

    def build(self) -> GraphSchema:
        node_type_ids = torch.tensor(
            [NODE_TYPE_IDS["feature"]] * len(FEATURE_NODE_NAMES)
            + [NODE_TYPE_IDS["function"]] * len(FUNCTION_NODE_NAMES)
            + [NODE_TYPE_IDS["domain"]] * len(DOMAIN_NODE_NAMES),
            dtype=torch.long,
        )
        level_ids = torch.tensor(
            [LEVEL_IDS["feature"]] * len(FEATURE_NODE_NAMES)
            + [LEVEL_IDS["function"]] * len(FUNCTION_NODE_NAMES)
            + [LEVEL_IDS["domain"]] * len(DOMAIN_NODE_NAMES),
            dtype=torch.long,
        )
        fc_edges: list[tuple[int, int]] = []
        fc_types: list[int] = []
        for relation_name in [
            "fc_prosodic",
            "fc_phonatory_articulatory",
            "fc_temporal_fluency",
            "fc_lexical",
            "fc_syntactic",
            "fc_discourse",
        ]:
            for source_name, target_name in self.ontology.fc_edges_by_relation[relation_name]:
                fc_edges.append((self.node_index[source_name], self.node_index[target_name]))
                fc_types.append(RELATION_IDS[relation_name])
        cd_edges = [(self.node_index[source], self.node_index[target]) for source, target in self.ontology.cd_edges]
        cd_types = [RELATION_IDS["cd_hierarchy"]] * len(cd_edges)
        self_loops = [(self.node_index[source], self.node_index[target]) for source, target in self.ontology.self_loops]
        self_types = [RELATION_IDS["self_loop"]] * len(self_loops)
        all_edges = fc_edges + cd_edges + self_loops
        edge_type = fc_types + cd_types + self_types
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
        fc_edge_mask = torch.zeros(len(all_edges), dtype=torch.bool)
        cd_edge_mask = torch.zeros(len(all_edges), dtype=torch.bool)
        self_loop_mask = torch.zeros(len(all_edges), dtype=torch.bool)
        fc_edge_mask[: len(fc_edges)] = True
        cd_edge_mask[len(fc_edges) : len(fc_edges) + len(cd_edges)] = True
        self_loop_mask[len(fc_edges) + len(cd_edges) :] = True
        return GraphSchema(
            node_names=list(self.ontology.node_names),
            node_type_ids=node_type_ids,
            level_ids=level_ids,
            edge_index=edge_index,
            edge_type=edge_type_tensor,
            fc_edge_mask=fc_edge_mask,
            cd_edge_mask=cd_edge_mask,
            self_loop_mask=self_loop_mask,
            relation_id_to_name={value: key for key, value in RELATION_IDS.items()},
            feature_node_indices=torch.arange(0, len(FEATURE_NODE_NAMES), dtype=torch.long),
            function_node_indices=torch.arange(len(FEATURE_NODE_NAMES), len(FEATURE_NODE_NAMES) + len(FUNCTION_NODE_NAMES), dtype=torch.long),
            domain_node_indices=torch.arange(len(FEATURE_NODE_NAMES) + len(FUNCTION_NODE_NAMES), len(NODE_NAMES), dtype=torch.long),
        )


def build_graph_schema(ontology: NeurolinguisticOntology) -> GraphSchema:
    return GraphSchemaBuilder(ontology).build()

