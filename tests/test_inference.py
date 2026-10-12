import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from pathlib import Path

from conftest import _write_wave
from speechcgt.data.schemas import SpeechSample
from speechcgt.features.extractor import SpeechFeatureExtractor
from speechcgt.features.normalizer import FeatureNormalizer
from speechcgt.graph.builder import NeurolinguisticGraphBuilder
from speechcgt.inference.pipeline import SpeechCGTInferenceEngine
from speechcgt.models.framework import SpeechCGTFramework
from speechcgt.training.calibrator import TemperatureScaler


def test_inference_report_and_export(toy_config, graph_schema, tmp_path):
    audio_path = tmp_path / "infer.wav"
    _write_wave(audio_path)
    model = SpeechCGTFramework(toy_config, graph_schema)
    extractor = SpeechFeatureExtractor(toy_config)
    normalizer = FeatureNormalizer(toy_config)
    sample = SpeechSample(
        sample_id="infer_sample",
        participant_id="infer_participant",
        audio_path=str(audio_path),
        transcript="the boy reaches and then the water spills",
        label=1,
        corpus_id=0,
        corpus_name="Pitt",
    )
    preprocessed = __import__("speechcgt.data.preprocessing", fromlist=["preprocess_sample"]).preprocess_sample(
        sample,
        target_sample_rate=toy_config["audio"]["target_sample_rate"],
        target_lufs=toy_config["audio"]["target_lufs"],
        transcript_config=toy_config["transcript"],
        audio_config=toy_config["audio"],
    )
    x_raw, mask = extractor.extract(preprocessed)
    stats = normalizer.fit_feature_stats(x_raw.unsqueeze(0), mask.unsqueeze(0), "Pitt", 0)
    scaler = TemperatureScaler(toy_config)
    scaler.temperature = 2.0
    engine = SpeechCGTInferenceEngine(model, graph_schema, stats, scaler, toy_config)
    report = engine.predict_sample(sample)
    output_path = tmp_path / "report.json"
    engine.export_report(report, output_path)
    assert output_path.exists()
    assert len(report.top_feature_to_function) <= toy_config["inference"]["top_k_fc"]
    assert len(report.top_function_to_domain) <= toy_config["inference"]["top_k_cd"]
    assert 0.0 <= report.ci_prob_calibrated <= 1.0
