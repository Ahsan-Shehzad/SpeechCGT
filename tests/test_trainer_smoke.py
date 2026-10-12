import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from conftest import make_loader, make_manifest_dataframe
from speechcgt.data.schemas import SplitManifest
from speechcgt.features.extractor import SpeechFeatureExtractor
from speechcgt.features.normalizer import FeatureNormalizer
from speechcgt.graph.builder import NeurolinguisticGraphBuilder
from speechcgt.models.framework import SpeechCGTFramework
from speechcgt.training.trainer import SpeechCGTTrainer


def test_trainer_smoke(toy_config, graph_schema, tmp_path):
    source_df = make_manifest_dataframe(tmp_path, "Pitt", [0, 1, 0, 1], "train")
    val_df = make_manifest_dataframe(tmp_path, "Pitt", [0, 1], "val")
    test_df = make_manifest_dataframe(tmp_path, "Pitt", [0, 1], "test")
    external_df = make_manifest_dataframe(tmp_path, "WLS", [0, 1], "ext")
    split = SplitManifest(
        source_corpus="Pitt",
        fold_id=0,
        train_sample_ids=source_df["sample_id"].tolist(),
        val_sample_ids=val_df["sample_id"].tolist(),
        test_sample_ids=test_df["sample_id"].tolist(),
        external_sample_ids=external_df["sample_id"].tolist(),
        train_participant_ids=source_df["participant_id"].tolist(),
        val_participant_ids=val_df["participant_id"].tolist(),
        test_participant_ids=test_df["participant_id"].tolist(),
        external_participant_ids=external_df["participant_id"].tolist(),
    )
    model = SpeechCGTFramework(toy_config, graph_schema)
    trainer = SpeechCGTTrainer(
        model,
        SpeechFeatureExtractor(toy_config),
        FeatureNormalizer(toy_config),
        NeurolinguisticGraphBuilder(graph_schema),
        toy_config,
    )
    result = trainer.fit(
        make_loader(source_df, 2),
        make_loader(val_df, 2),
        make_loader(test_df, 2),
        make_loader(external_df, 2),
        split,
    )
    assert "checkpoint_path" in result
    assert result["temperature"] is not None
