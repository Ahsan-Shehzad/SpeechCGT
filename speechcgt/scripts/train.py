from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.datasets import SpeechCGTDataset
from ..data.manifests import build_group_stratified_splits, load_corpus_manifest
from ..data.collate import collate_speech_samples
from ..graph.builder import NeurolinguisticGraphBuilder
from ..graph.ontology import build_default_ontology
from ..graph.schema import build_graph_schema
from ..features.extractor import SpeechFeatureExtractor
from ..features.normalizer import FeatureNormalizer
from ..models.framework import SpeechCGTFramework
from ..training.trainer import SpeechCGTTrainer
from ..utils.io import ensure_dir, save_json


def _make_loader(dataframe, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        SpeechCGTDataset(dataframe),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_speech_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SpeechCGT framework.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--fold", type=int, default=None, help="Optional single fold to train.")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest_dir = Path(config["data"]["manifest_dir"])
    source_key = config["experiment"]["source_corpus"].lower()
    source_manifest = load_corpus_manifest(manifest_dir / f"{source_key}.csv")
    external_manifest_path = manifest_dir / "wls.csv"
    if external_manifest_path.exists():
        external_manifest = load_corpus_manifest(external_manifest_path)
        combined_manifest = pd.concat([source_manifest, external_manifest], ignore_index=True)
    else:
        combined_manifest = source_manifest.copy()
    splits = build_group_stratified_splits(
        combined_manifest,
        num_folds=config["evaluation"]["num_folds"],
        group_key=config["data"]["participant_id_key"],
        label_key=config["data"]["label_key"],
        train_ratio=config["evaluation"]["train_ratio"],
        val_ratio=config["evaluation"]["val_ratio"],
        test_ratio=config["evaluation"]["test_ratio"],
        seed=config["experiment"]["seed"],
        source_corpus=config["experiment"]["source_corpus"],
    )
    selected_splits = [splits[args.fold]] if args.fold is not None else splits
    summaries = []
    for split in selected_splits:
        ontology = build_default_ontology()
        graph_schema = build_graph_schema(ontology)
        feature_extractor = SpeechFeatureExtractor(config)
        normalizer = FeatureNormalizer(config)
        graph_builder = NeurolinguisticGraphBuilder(graph_schema)
        model = SpeechCGTFramework(config, graph_schema)
        trainer = SpeechCGTTrainer(model, feature_extractor, normalizer, graph_builder, config)
        train_df = combined_manifest[combined_manifest["sample_id"].astype(str).isin(split.train_sample_ids)]
        val_df = combined_manifest[combined_manifest["sample_id"].astype(str).isin(split.val_sample_ids)]
        test_df = combined_manifest[combined_manifest["sample_id"].astype(str).isin(split.test_sample_ids)]
        external_df = combined_manifest[combined_manifest["sample_id"].astype(str).isin(split.external_sample_ids)]
        result = trainer.fit(
            _make_loader(train_df, config["training"]["batch_size"], True),
            _make_loader(val_df, config["training"]["batch_size"], False),
            _make_loader(test_df, config["training"]["batch_size"], False),
            _make_loader(external_df, config["training"]["batch_size"], False),
            split,
        )
        summaries.append(result)
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    save_json(summaries, output_dir / "training_runs.json")


if __name__ == "__main__":
    main()
