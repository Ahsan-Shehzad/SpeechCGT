import pytest

pytest.importorskip("torch")

import torch

from speechcgt.data.preprocessing import preprocess_sample
from speechcgt.data.schemas import SpeechSample
from speechcgt.features.extractor import SpeechFeatureExtractor
from speechcgt.features.normalizer import FeatureNormalizer


def test_feature_extraction_and_normalization(toy_config, tmp_path):
    from conftest import _write_wave

    audio_path = tmp_path / "feature.wav"
    _write_wave(audio_path)
    sample = SpeechSample(
        sample_id="s1",
        participant_id="p1",
        audio_path=str(audio_path),
        transcript="The boy is taking cookies and then the water spills.",
        label=1,
        corpus_id=0,
        corpus_name="Pitt",
    )
    preprocessed = preprocess_sample(
        sample,
        target_sample_rate=toy_config["audio"]["target_sample_rate"],
        target_lufs=toy_config["audio"]["target_lufs"],
        transcript_config=toy_config["transcript"],
        audio_config=toy_config["audio"],
    )
    extractor = SpeechFeatureExtractor(toy_config)
    x_raw, mask = extractor.extract(preprocessed)
    assert tuple(x_raw.shape) == (24,)
    assert tuple(mask.shape) == (24,)
    normalizer = FeatureNormalizer(toy_config)
    stats = normalizer.fit_feature_stats(x_raw.unsqueeze(0), mask.unsqueeze(0), "Pitt", 0)
    x_hat, out_mask = normalizer.transform_features(x_raw.unsqueeze(0), mask.unsqueeze(0), stats)
    assert tuple(x_hat.shape) == (1, 24)
    assert torch.equal(mask.unsqueeze(0), out_mask)
