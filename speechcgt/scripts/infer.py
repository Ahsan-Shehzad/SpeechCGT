from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_config
from ..constants import CORPUS_IDS
from ..data.schemas import SpeechSample
from ..features.normalizer import FeatureStats
from ..graph.ontology import build_default_ontology
from ..graph.schema import build_graph_schema
from ..inference.pipeline import SpeechCGTInferenceEngine
from ..models.framework import SpeechCGTFramework
from ..training.calibrator import TemperatureScaler
from ..utils.io import load_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SpeechCGT inference for a single sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--transcript", default=None)
    parser.add_argument("--transcript-path", default=None)
    parser.add_argument("--sample-id", default="sample")
    parser.add_argument("--participant-id", default="participant")
    parser.add_argument("--corpus-name", default="Pitt")
    parser.add_argument("--label", type=int, default=None)
    parser.add_argument("--config", required=False)
    parser.add_argument("--output", default=None)
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
    transcript = args.transcript
    if transcript is None and args.transcript_path:
        transcript = Path(args.transcript_path).read_text(encoding="utf-8", errors="ignore")
    transcript = transcript or ""
    sample = SpeechSample(
        sample_id=args.sample_id,
        participant_id=args.participant_id,
        audio_path=args.audio_path,
        transcript=transcript,
        label=args.label,
        corpus_id=CORPUS_IDS.get(args.corpus_name, CORPUS_IDS["Pitt"]),
        corpus_name=args.corpus_name,
    )
    engine = SpeechCGTInferenceEngine(model, graph_schema, feature_stats, scaler, config)
    report = engine.predict_sample(sample)
    if args.output:
        engine.export_report(report, args.output)
    print(report.to_dict())


if __name__ == "__main__":
    main()

