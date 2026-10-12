from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
try:
    import soundfile as sf
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    sf = None

try:
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    DataLoader = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _write_wave(path: Path, sample_rate: int = 16000, seconds: float = 1.5) -> None:
    if sf is None:
        pytest.skip("soundfile is not installed")
    time = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 220 * time)
    tone[int(0.4 * sample_rate) : int(0.6 * sample_rate)] = 0.0
    sf.write(path, tone.astype(np.float32), sample_rate)


def make_manifest_dataframe(tmp_path: Path, corpus_name: str, labels: list[int], prefix: str) -> pd.DataFrame:
    records = []
    for index, label in enumerate(labels):
        audio_path = tmp_path / f"{prefix}_{index}.wav"
        _write_wave(audio_path)
        records.append(
            {
                "sample_id": f"{prefix}_sample_{index}",
                "participant_id": f"{prefix}_participant_{index}",
                "audio_path": str(audio_path),
                "transcript": "the boy is reaching for cookies and then the girl watches the water overflow",
                "label": label,
                "corpus_id": {"Pitt": 0, "Delaware": 1, "WLS": 2}[corpus_name],
                "corpus_name": corpus_name,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def toy_config(tmp_path: Path):
    from speechcgt.config import load_config

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    config["experiment"]["device"] = "cpu"
    config["experiment"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 2
    config["optimization"]["warmup_epochs"] = 1
    return config


@pytest.fixture
def graph_schema():
    pytest.importorskip("torch")
    from speechcgt.graph.ontology import build_default_ontology
    from speechcgt.graph.schema import build_graph_schema

    return build_graph_schema(build_default_ontology())


@pytest.fixture
def tiny_manifest(tmp_path: Path):
    source_df = make_manifest_dataframe(tmp_path, "Pitt", [0, 1, 0, 1], "src")
    external_df = make_manifest_dataframe(tmp_path, "WLS", [0, 1], "ext")
    return pd.concat([source_df, external_df], ignore_index=True)


def make_loader(df: pd.DataFrame, batch_size: int = 2) -> DataLoader:
    if DataLoader is None:
        pytest.skip("torch is not installed")
    from speechcgt.data.collate import collate_speech_samples
    from speechcgt.data.datasets import SpeechCGTDataset

    return DataLoader(SpeechCGTDataset(df), batch_size=batch_size, shuffle=False, collate_fn=collate_speech_samples)
