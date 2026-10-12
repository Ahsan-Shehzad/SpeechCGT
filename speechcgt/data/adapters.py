from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..config import load_config
from ..constants import CORPUS_IDS
from ..utils.io import ensure_dir
from .schemas import RawCorpusRecord


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported metadata format: {path}")


def _resolve_path(raw_root: Path, value: str | None) -> str:
    if not value:
        return ""
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((raw_root / candidate).resolve())


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "mci", "ci", "impaired"}


class CorpusAdapter(ABC):
    corpus_name: str
    corpus_id: int

    def __init__(self, corpus_name: str) -> None:
        self.corpus_name = corpus_name
        self.corpus_id = CORPUS_IDS[corpus_name]

    def _load_metadata(self, config: dict[str, Any]) -> pd.DataFrame | None:
        metadata_path = config.get("metadata_path")
        if metadata_path:
            return _read_table(metadata_path)
        return None

    def _discover_from_metadata(self, raw_root: Path, config: dict[str, Any]) -> list[RawCorpusRecord]:
        metadata_df = self._load_metadata(config)
        if metadata_df is None:
            return []
        sample_id_column = config.get("sample_id_column", "sample_id")
        participant_id_column = config.get("participant_id_column", "participant_id")
        audio_path_column = config.get("audio_path_column", "audio_path")
        transcript_path_column = config.get("transcript_path_column", "transcript_path")
        transcript_text_column = config.get("transcript_text_column", "transcript")
        label_column = config.get("label_column", "label")
        records: list[RawCorpusRecord] = []
        for row in metadata_df.to_dict("records"):
            transcript_text = row.get(transcript_text_column)
            transcript_path = row.get(transcript_path_column)
            record = RawCorpusRecord(
                sample_id=str(row[sample_id_column]),
                participant_id=str(row[participant_id_column]),
                audio_path=_resolve_path(raw_root, row.get(audio_path_column)),
                transcript_path=_resolve_path(raw_root, transcript_path) if transcript_path else None,
                transcript_text=str(transcript_text) if transcript_text is not None and pd.notna(transcript_text) else None,
                label=self._map_label(row.get(label_column), row),
                corpus_id=self.corpus_id,
                corpus_name=self.corpus_name,
                metadata=row,
            )
            records.append(record)
        return records

    def _discover_from_files(self, raw_root: Path, config: dict[str, Any]) -> list[RawCorpusRecord]:
        audio_glob = config.get("audio_glob", "**/*.wav")
        transcript_glob = config.get("transcript_glob", "**/*.cha")
        transcript_files = {path.stem: path for path in raw_root.glob(transcript_glob)}
        records: list[RawCorpusRecord] = []
        for audio_path in raw_root.glob(audio_glob):
            sample_id = audio_path.stem
            transcript_path = transcript_files.get(sample_id)
            records.append(
                RawCorpusRecord(
                    sample_id=sample_id,
                    participant_id=sample_id.split("_")[0],
                    audio_path=str(audio_path.resolve()),
                    transcript_path=str(transcript_path.resolve()) if transcript_path else None,
                    label=None,
                    corpus_id=self.corpus_id,
                    corpus_name=self.corpus_name,
                )
            )
        return records

    def discover_records(self, raw_root: str | Path, config: dict[str, Any]) -> list[RawCorpusRecord]:
        raw_root = Path(raw_root)
        records = self._discover_from_metadata(raw_root, config)
        if records:
            return records
        return self._discover_from_files(raw_root, config)

    def load_transcript(self, record: RawCorpusRecord) -> str:
        if record.transcript_text is not None:
            return record.transcript_text
        if record.transcript_path:
            path = Path(record.transcript_path)
            if path.exists():
                return path.read_text(encoding="utf-8", errors="ignore")
        return ""

    def build_manifest(self, output_path: str | Path, config: dict[str, Any]) -> pd.DataFrame:
        raw_root = Path(config.get("raw_root") or config.get("root") or ".")
        records = self.discover_records(raw_root, config)
        rows = []
        for record in records:
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "participant_id": record.participant_id,
                    "audio_path": record.audio_path,
                    "transcript": self.load_transcript(record),
                    "label": record.label,
                    "corpus_id": self.corpus_id,
                    "corpus_name": self.corpus_name,
                }
            )
        manifest_df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(output_path, index=False)
        return manifest_df

    @abstractmethod
    def _map_label(self, value: Any, row: dict[str, Any]) -> int | None:
        raise NotImplementedError


class PittCorpusAdapter(CorpusAdapter):
    def __init__(self) -> None:
        super().__init__("Pitt")

    def _map_label(self, value: Any, row: dict[str, Any]) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip().lower()
        if text in {"0", "cn", "control", "controls", "healthy"}:
            return 0
        if any(token in text for token in {"1", "ci", "ad", "probable", "possible", "alz"}):
            return 1
        return None


class DelawareCorpusAdapter(CorpusAdapter):
    def __init__(self) -> None:
        super().__init__("Delaware")

    def _map_label(self, value: Any, row: dict[str, Any]) -> int | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip().lower()
        if text in {"0", "cn", "control", "neurotypical", "healthy"}:
            return 0
        if any(token in text for token in {"1", "ci", "mci", "impaired", "ad"}):
            return 1
        return None


class WLSCorpusAdapter(CorpusAdapter):
    def __init__(self) -> None:
        super().__init__("WLS")

    def _map_label(self, value: Any, row: dict[str, Any]) -> int | None:
        diagnosis_key = row.get("diagnosis_2020", row.get("label_2020", value))
        prior_key = row.get("impairment_2011", row.get("prior_impairment_2011"))
        if diagnosis_key is None and prior_key is None:
            return None
        diagnosis = "" if diagnosis_key is None else str(diagnosis_key).strip().lower()
        prior_impairment = _as_bool(prior_key)
        if diagnosis in {"normal", "cn", "control", "healthy"} and not prior_impairment:
            return 0
        if prior_impairment or any(token in diagnosis for token in {"mci", "dementia", "impaired", "ci"}):
            return 1
        return None


ADAPTERS = {
    "pitt": PittCorpusAdapter,
    "delaware": DelawareCorpusAdapter,
    "wls": WLSCorpusAdapter,
}


def build_manifests(config_path: str | None = None) -> dict[str, str]:
    if config_path is None:
        parser = argparse.ArgumentParser(description="Build normalized SpeechCGT manifests.")
        parser.add_argument("--config", required=True, help="Path to YAML config.")
        args = parser.parse_args()
        config_path = args.config
    config = load_config(config_path)
    manifest_dir = ensure_dir(config["data"]["manifest_dir"])
    built_paths: dict[str, str] = {}
    for adapter_name, adapter_cls in ADAPTERS.items():
        adapter_config = config["data"]["adapters"].get(adapter_name, {})
        raw_root = adapter_config.get("raw_root") or adapter_config.get("root")
        metadata_path = adapter_config.get("metadata_path")
        if not raw_root and not metadata_path:
            continue
        adapter = adapter_cls()
        output_path = manifest_dir / f"{adapter_name}.csv"
        adapter.build_manifest(output_path, adapter_config)
        built_paths[adapter_name] = str(output_path)
    return built_paths

