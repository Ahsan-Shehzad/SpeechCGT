from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from ..config import load_config
from ..data.collate import collate_speech_samples
from ..data.datasets import SpeechCGTDataset
from ..data.manifests import load_corpus_manifest
from ..features.extractor import SpeechFeatureExtractor
from ..features.normalizer import FeatureNormalizer, FeatureStats
from ..graph.builder import NeurolinguisticGraphBuilder
from ..graph.ontology import build_default_ontology
from ..graph.schema import build_graph_schema
from ..models.framework import SpeechCGTFramework
from ..training.calibrator import TemperatureScaler
from ..training.trainer import evaluate_split
from ..utils.io import load_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SpeechCGT checkpoint on a labeled manifest.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=False)
    args = parser.parse_args()
    payload = load_torch(args.checkpoint)
    config = load_config(args.config) if args.config else payload["config"]
    ontology = build_default_ontology()
    graph_schema = build_graph_schema(ontology)
    model = SpeechCGTFramework(config, graph_schema)
    model.load_state_dict(payload["model_state_dict"])
    feature_stats = FeatureStats.from_dict(payload["feature_stats"])
    scaler = None
    if payload.get("temperature") is not None:
        scaler = TemperatureScaler(config)
        scaler.temperature = float(payload["temperature"])
    loader = DataLoader(
        SpeechCGTDataset(load_corpus_manifest(args.manifest)),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_speech_samples,
    )
    metrics = evaluate_split(
        model,
        loader,
        SpeechFeatureExtractor(config),
        FeatureNormalizer(config),
        NeurolinguisticGraphBuilder(graph_schema),
        config["calibration"]["threshold"],
        scaler,
        next(model.parameters()).device,
        config,
        feature_stats,
    )
    print(metrics)


if __name__ == "__main__":
    main()

