from pathlib import Path

import pandas as pd

from speechcgt.data.adapters import DelawareCorpusAdapter, PittCorpusAdapter, WLSCorpusAdapter


def test_adapters_build_manifest(tmp_path):
    transcript_path = tmp_path / "sample.cha"
    transcript_path.write_text("The boy takes cookies.", encoding="utf-8")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")
    metadata = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "participant_id": "p1",
                "audio_path": "sample.wav",
                "transcript_path": "sample.cha",
                "label": "control",
            }
        ]
    )
    metadata_path = tmp_path / "pitt.csv"
    metadata.to_csv(metadata_path, index=False)
    adapter = PittCorpusAdapter()
    manifest = adapter.build_manifest(tmp_path / "manifest.csv", {"raw_root": str(tmp_path), "metadata_path": str(metadata_path)})
    assert manifest.loc[0, "label"] == 0
    assert manifest.loc[0, "corpus_name"] == "Pitt"


def test_wls_adapter_conservative_label(tmp_path):
    transcript_path = tmp_path / "sample.cha"
    transcript_path.write_text("Then the water spills.", encoding="utf-8")
    metadata = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "participant_id": "p1",
                "audio_path": "missing.wav",
                "transcript_path": "sample.cha",
                "diagnosis_2020": "normal",
                "impairment_2011": False,
            },
            {
                "sample_id": "s2",
                "participant_id": "p2",
                "audio_path": "missing.wav",
                "transcript_path": "sample.cha",
                "diagnosis_2020": "MCI",
                "impairment_2011": False,
            },
        ]
    )
    metadata_path = tmp_path / "wls.csv"
    metadata.to_csv(metadata_path, index=False)
    adapter = WLSCorpusAdapter()
    manifest = adapter.build_manifest(tmp_path / "manifest.csv", {"raw_root": str(tmp_path), "metadata_path": str(metadata_path)})
    assert manifest["label"].tolist() == [0, 1]

