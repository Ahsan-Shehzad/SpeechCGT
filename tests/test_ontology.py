import pytest

pytest.importorskip("torch")

from speechcgt.constants import FUNCTION_NODE_NAMES, NODE_NAMES, RELATION_IDS
from speechcgt.graph.ontology import build_default_ontology
from speechcgt.graph.schema import build_graph_schema


def test_ontology_counts_and_order():
    ontology = build_default_ontology()
    schema = build_graph_schema(ontology)
    assert len(ontology.feature_nodes) == 24
    assert len(ontology.function_nodes) == 10
    assert len(ontology.domain_nodes) == 4
    assert len(schema.node_names) == 38
    assert schema.node_names == NODE_NAMES
    assert len(RELATION_IDS) == 8
    assert schema.function_node_indices.tolist() == list(range(24, 34))
    assert FUNCTION_NODE_NAMES[0] == "lexical_retrieval"
