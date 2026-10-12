from __future__ import annotations

from dataclasses import dataclass

from ..constants import (
    DOMAIN_NODE_NAMES,
    FEATURE_NODE_NAMES,
    FUNCTION_NODE_NAMES,
    LEVEL_IDS,
    NODE_NAMES,
    NODE_TYPE_IDS,
    RELATION_IDS,
)


@dataclass(frozen=True)
class NeurolinguisticOntology:
    feature_nodes: list[str]
    function_nodes: list[str]
    domain_nodes: list[str]
    relation_ids: dict[str, int]
    fc_edges_by_relation: dict[str, list[tuple[str, str]]]
    cd_edges: list[tuple[str, str]]
    self_loops: list[tuple[str, str]]

    @property
    def node_names(self) -> list[str]:
        return self.feature_nodes + self.function_nodes + self.domain_nodes


def build_default_ontology() -> NeurolinguisticOntology:
    fc_edges_by_relation = {
        "fc_prosodic": [
            (feature, target)
            for feature in ["pitch_variability", "pitch_range", "loudness_variability", "prosodic_reset"]
            for target in ["prosodic_control", "attention_maintenance"]
        ],
        "fc_phonatory_articulatory": [
            (feature, target)
            for feature in ["jitter_shimmer", "harmonics_to_noise", "cepstral_prominence", "formant_stability"]
            for target in ["articulatory_control", "self_monitoring"]
        ],
        "fc_temporal_fluency": [
            (feature, target)
            for feature in [
                "speech_rate",
                "articulation_rate",
                "silence_ratio",
                "mean_pause_duration",
                "pause_frequency",
                "filled_pause_rate",
                "self_repair_rate",
                "utterance_length",
            ]
            for target in ["verbal_fluency", "attention_maintenance", "self_monitoring"]
        ],
        "fc_lexical": [
            (feature, target)
            for feature in ["lexical_richness", "content_word_density", "pronoun_dependency"]
            for target in ["lexical_retrieval", "semantic_access"]
        ],
        "fc_syntactic": [
            (feature, target)
            for feature in ["syntactic_depth", "clause_linking_ratio"]
            for target in ["syntactic_planning", "working_memory_buffering"]
        ],
        "fc_discourse": [
            (feature, target)
            for feature in ["semantic_coherence", "information_unit_density", "discourse_sequencing"]
            for target in ["semantic_access", "discourse_organization", "working_memory_buffering"]
        ],
    }
    cd_edges = [
        ("lexical_retrieval", "language"),
        ("semantic_access", "language"),
        ("syntactic_planning", "language"),
        ("discourse_organization", "language"),
        ("verbal_fluency", "language"),
        ("articulatory_control", "language"),
        ("prosodic_control", "language"),
        ("semantic_access", "semantic_memory"),
        ("discourse_organization", "executive_control"),
        ("verbal_fluency", "executive_control"),
        ("self_monitoring", "executive_control"),
        ("prosodic_control", "attention_working_memory"),
        ("working_memory_buffering", "attention_working_memory"),
        ("attention_maintenance", "attention_working_memory"),
    ]
    self_loops = [(node, node) for node in NODE_NAMES]
    return NeurolinguisticOntology(
        feature_nodes=list(FEATURE_NODE_NAMES),
        function_nodes=list(FUNCTION_NODE_NAMES),
        domain_nodes=list(DOMAIN_NODE_NAMES),
        relation_ids=dict(RELATION_IDS),
        fc_edges_by_relation=fc_edges_by_relation,
        cd_edges=cd_edges,
        self_loops=self_loops,
    )

