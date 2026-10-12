from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Batch

from ..constants import DOMAIN_NODE_NAMES, FEATURE_NODE_NAMES, FUNCTION_NODE_NAMES
from ..data.preprocessing import preprocess_sample
from ..data.schemas import SpeechSample
from ..features.extractor import SpeechFeatureExtractor
from ..features.normalizer import FeatureNormalizer, FeatureStats
from ..graph.builder import NeurolinguisticGraphBuilder
from ..graph.schema import GraphSchema
from ..models.framework import SpeechCGTFramework
from ..training.calibrator import TemperatureScaler
from ..utils.io import load_torch
from .reporting import (
    InferenceReport,
    export_cognitive_map_json,
    rank_feature_to_function_paths,
    rank_function_to_domain_paths,
)


class CognitiveMapReporter:
    def __init__(self, graph_schema: GraphSchema, config: dict[str, Any]) -> None:
        self.graph_schema = graph_schema
        self.config = config

    def build_report(
        self,
        sample: SpeechSample,
        ci_logit: torch.FloatTensor,
        ci_prob_raw: torch.FloatTensor,
        ci_prob_calibrated: torch.FloatTensor,
        output,
    ) -> InferenceReport:
        top_fc = rank_feature_to_function_paths(
            output.path_fc[0],
            self.config["inference"]["top_k_fc"],
            FEATURE_NODE_NAMES,
            FUNCTION_NODE_NAMES,
        )
        top_cd = rank_function_to_domain_paths(
            output.path_cd[0],
            self.config["inference"]["top_k_cd"],
            FUNCTION_NODE_NAMES,
            DOMAIN_NODE_NAMES,
        )
        predicted_label = int(ci_prob_calibrated.item() >= self.config["calibration"]["threshold"])
        return InferenceReport(
            sample_id=sample.sample_id,
            participant_id=sample.participant_id,
            ci_logit=float(ci_logit.item()),
            ci_prob_raw=float(ci_prob_raw.item()),
            ci_prob_calibrated=float(ci_prob_calibrated.item()),
            predicted_label=predicted_label,
            feature_scores=output.feature_scores[0].detach().cpu().tolist(),
            function_scores=output.function_scores[0].detach().cpu().tolist(),
            domain_scores=output.domain_scores[0].detach().cpu().tolist(),
            top_feature_to_function=top_fc,
            top_function_to_domain=top_cd,
        )


class SpeechCGTInferenceEngine:
    def __init__(
        self,
        model: SpeechCGTFramework,
        graph_schema: GraphSchema,
        feature_stats: FeatureStats,
        temperature_scaler: TemperatureScaler | None,
        config: dict[str, Any],
        checkpoint: str | dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.graph_schema = graph_schema
        self.feature_stats = feature_stats
        self.temperature_scaler = temperature_scaler
        self.config = config
        self.feature_extractor = SpeechFeatureExtractor(config)
        self.normalizer = FeatureNormalizer(config)
        self.graph_builder = NeurolinguisticGraphBuilder(graph_schema)
        self.reporter = CognitiveMapReporter(graph_schema, config)
        device_name = config["experiment"].get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.model.eval()
        if checkpoint is not None:
            payload = load_torch(checkpoint) if isinstance(checkpoint, (str, Path)) else checkpoint
            self.model.load_state_dict(payload["model_state_dict"])

    @torch.no_grad()
    def predict_sample(self, sample: SpeechSample) -> InferenceReport:
        preprocessed = preprocess_sample(
            sample,
            target_sample_rate=self.config["audio"]["target_sample_rate"],
            target_lufs=self.config["audio"]["target_lufs"],
            transcript_config=self.config["transcript"],
            audio_config=self.config["audio"],
        )
        x_raw, feature_mask = self.feature_extractor.extract(preprocessed)
        x_hat, feature_mask = self.normalizer.transform_features(x_raw.unsqueeze(0), feature_mask.unsqueeze(0), self.feature_stats)
        graph = self.graph_builder.build_subject_graph(x_hat[0], feature_mask[0], sample.label)
        batch = Batch.from_data_list([graph]).to(self.device)
        output = self.model(batch, x_hat.to(self.device), feature_mask.to(self.device))
        raw_prob = output.ci_prob.detach().cpu()
        calibrated_prob = (
            self.temperature_scaler.transform(output.ci_logit.detach().cpu())
            if self.temperature_scaler is not None
            else raw_prob
        )
        return self.reporter.build_report(sample, output.ci_logit.detach().cpu(), raw_prob, calibrated_prob, output)

    def export_report(self, report: InferenceReport, output_path: str | Path) -> None:
        export_cognitive_map_json(report, output_path)
