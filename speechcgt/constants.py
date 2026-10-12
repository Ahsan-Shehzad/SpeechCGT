from __future__ import annotations

from copy import deepcopy

FEATURE_NODE_NAMES = [
    "pitch_variability",
    "pitch_range",
    "loudness_variability",
    "speech_rate",
    "articulation_rate",
    "prosodic_reset",
    "jitter_shimmer",
    "harmonics_to_noise",
    "cepstral_prominence",
    "formant_stability",
    "silence_ratio",
    "mean_pause_duration",
    "pause_frequency",
    "filled_pause_rate",
    "self_repair_rate",
    "utterance_length",
    "lexical_richness",
    "content_word_density",
    "pronoun_dependency",
    "syntactic_depth",
    "clause_linking_ratio",
    "semantic_coherence",
    "information_unit_density",
    "discourse_sequencing",
]

FUNCTION_NODE_NAMES = [
    "lexical_retrieval",
    "semantic_access",
    "syntactic_planning",
    "discourse_organization",
    "verbal_fluency",
    "articulatory_control",
    "prosodic_control",
    "working_memory_buffering",
    "attention_maintenance",
    "self_monitoring",
]

DOMAIN_NODE_NAMES = [
    "language",
    "semantic_memory",
    "executive_control",
    "attention_working_memory",
]

NODE_NAMES = FEATURE_NODE_NAMES + FUNCTION_NODE_NAMES + DOMAIN_NODE_NAMES

NODE_TYPE_IDS = {
    "feature": 0,
    "function": 1,
    "domain": 2,
}

LEVEL_IDS = {
    "feature": 0,
    "function": 1,
    "domain": 2,
}

CORPUS_IDS = {
    "Pitt": 0,
    "Delaware": 1,
    "WLS": 2,
}

RELATION_IDS = {
    "self_loop": 0,
    "fc_prosodic": 1,
    "fc_phonatory_articulatory": 2,
    "fc_temporal_fluency": 3,
    "fc_lexical": 4,
    "fc_syntactic": 5,
    "fc_discourse": 6,
    "cd_hierarchy": 7,
}

RELATION_NAMES = {value: key for key, value in RELATION_IDS.items()}

ACOUSTIC_FEATURE_NAMES = [
    "pitch_variability",
    "pitch_range",
    "loudness_variability",
    "speech_rate",
    "articulation_rate",
    "prosodic_reset",
    "jitter_shimmer",
    "harmonics_to_noise",
    "cepstral_prominence",
    "formant_stability",
    "silence_ratio",
    "mean_pause_duration",
    "pause_frequency",
]

TRANSCRIPT_FEATURE_NAMES = [
    "filled_pause_rate",
    "self_repair_rate",
    "utterance_length",
    "lexical_richness",
    "content_word_density",
    "pronoun_dependency",
    "syntactic_depth",
    "clause_linking_ratio",
    "semantic_coherence",
    "information_unit_density",
    "discourse_sequencing",
]

ACOUSTIC_FEATURE_INDICES = [FEATURE_NODE_NAMES.index(name) for name in ACOUSTIC_FEATURE_NAMES]
TRANSCRIPT_FEATURE_INDICES = [FEATURE_NODE_NAMES.index(name) for name in TRANSCRIPT_FEATURE_NAMES]

FILLER_TOKENS = {
    "uh",
    "um",
    "erm",
    "hmm",
    "mm",
    "mmm",
    "uhh",
    "umm",
}

SELF_REPAIR_MARKERS = {
    "--",
    "...",
    "i mean",
    "sorry",
    "no",
    "rather",
    "or",
}

SEQUENCING_CONNECTIVES = {
    "then",
    "next",
    "after",
    "afterward",
    "afterwards",
    "before",
    "while",
    "when",
    "finally",
    "first",
    "second",
    "third",
    "later",
    "meanwhile",
}

COOKIE_THEFT_INFORMATION_UNITS = {
    "boy",
    "girl",
    "mother",
    "woman",
    "kitchen",
    "sink",
    "stool",
    "cookie",
    "cookies",
    "jar",
    "plate",
    "dish",
    "water",
    "window",
    "curtain",
    "outside",
    "washing",
    "drying",
    "falling",
    "reaching",
    "stealing",
    "overflowing",
}

CONTENT_POS_TAGS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
PRONOUN_POS_TAGS = {"PRON"}
CLAUSE_LINKING_DEPS = {"mark", "cc", "advcl", "ccomp", "xcomp", "acl", "relcl", "conj"}

DEFAULT_CONFIG = {
    "experiment": {
        "name": "speechcgt",
        "source_corpus": "Pitt",
        "external_corpus": "WLS",
        "seed": 42,
        "device": "cuda",
        "output_dir": "outputs/speechcgt",
    },
    "data": {
        "manifest_dir": "speechcgt/data/manifests",
        "raw_data_dir": "speechcgt/data/raw",
        "processed_data_dir": "speechcgt/data/processed",
        "sample_id_key": "sample_id",
        "participant_id_key": "participant_id",
        "label_key": "label",
        "audio_path_key": "audio_path",
        "transcript_key": "transcript",
        "corpus_name_key": "corpus_name",
        "adapters": {
            "pitt": {
                "metadata_path": None,
                "audio_glob": "**/*.wav",
                "transcript_glob": "**/*.cha",
            },
            "delaware": {
                "metadata_path": None,
                "audio_glob": "**/*.wav",
                "transcript_glob": "**/*.cha",
            },
            "wls": {
                "metadata_path": None,
                "audio_glob": "**/*.wav",
                "transcript_glob": "**/*.cha",
                "label_2020_column": "diagnosis_2020",
                "prior_impairment_2011_column": "impairment_2011",
            },
        },
    },
    "audio": {
        "target_sample_rate": 16000,
        "mono": True,
        "dtype": "float32",
        "target_lufs": -23.0,
        "frame_length": 512,
        "hop_length": 160,
        "pause_threshold_db": -40.0,
        "min_pause_seconds": 0.2,
    },
    "transcript": {
        "lowercase": True,
        "normalize_punctuation": True,
        "remove_chat_control_symbols": True,
        "preserve_filled_pauses": True,
        "preserve_disfluency_markers": True,
        "spacy_model": "en_core_web_sm",
        "stanza_lang": "en",
        "mattr_window": 20,
    },
    "features": {
        "ordered_feature_names": FEATURE_NODE_NAMES,
        "acoustic_feature_names": ACOUSTIC_FEATURE_NAMES,
        "transcript_feature_names": TRANSCRIPT_FEATURE_NAMES,
        "normalization_epsilon": 1e-8,
    },
    "graph": {
        "feature_node_names": FEATURE_NODE_NAMES,
        "function_node_names": FUNCTION_NODE_NAMES,
        "domain_node_names": DOMAIN_NODE_NAMES,
        "relation_names": [RELATION_NAMES[idx] for idx in sorted(RELATION_NAMES)],
        "include_self_loops": True,
    },
    "model": {
        "hidden_dim": 128,
        "num_heads": 8,
        "head_dim": 16,
        "num_layers": 4,
        "ffn_dim": 512,
        "patient_summary_dim": 32,
        "dropout": 0.2,
        "activation": "gelu",
        "prenorm": True,
        "gate_lambda": 1.0,
        "gate_epsilon": 1e-6,
    },
    "training": {
        "batch_size": 16,
        "epochs": 150,
        "gradient_clip_norm": 1.0,
        "checkpoint_metric": "f1",
        "checkpoint_tie_break_metric": "accuracy",
    },
    "optimization": {
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": "linear_warmup_cosine_decay",
        "warmup_epochs": 10,
    },
    "augmentation": {
        "modality_dropout_p": 0.1,
        "gaussian_noise_std": 0.01,
        "acoustic_dropout_group": ACOUSTIC_FEATURE_NAMES,
        "transcript_dropout_group": TRANSCRIPT_FEATURE_NAMES,
    },
    "losses": {
        "classification_loss": "bce_with_logits",
        "consistency_weight": 0.2,
        "gate_sparsity_weight": 1e-4,
        "pos_weight_mode": "fit_on_train_partition",
    },
    "evaluation": {
        "num_folds": 5,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "positive_label": 1,
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "holdout_wls": True,
        "robustness_random_feature_group_mask_rate": 0.2,
    },
    "calibration": {
        "enabled": True,
        "method": "temperature_scaling",
        "threshold": 0.5,
        "fit_split": "validation",
        "max_iter": 200,
    },
    "inference": {
        "top_k_fc": 10,
        "top_k_cd": 6,
        "export_json": True,
        "return_raw_probability": True,
    },
    "logging": {
        "save_split_manifests": True,
        "save_feature_stats": True,
        "save_graph_schema": True,
        "save_temperature": True,
        "save_parameter_count": True,
        "save_runtime_logs": True,
    },
}


def get_default_config() -> dict:
    return deepcopy(DEFAULT_CONFIG)

