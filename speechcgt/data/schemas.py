from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class SpeechSample:
    sample_id: str
    participant_id: str
    audio_path: str
    transcript: str
    label: Optional[int]
    corpus_id: int
    corpus_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedSample:
    sample: SpeechSample
    waveform: np.ndarray
    sample_rate: int
    cleaned_transcript: str
    utterances: list[str] = field(default_factory=list)


@dataclass
class SplitManifest:
    source_corpus: str
    fold_id: int
    train_sample_ids: list[str]
    val_sample_ids: list[str]
    test_sample_ids: list[str]
    external_sample_ids: list[str]
    train_participant_ids: list[str]
    val_participant_ids: list[str]
    test_participant_ids: list[str]
    external_participant_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RawCorpusRecord:
    sample_id: str
    participant_id: str
    audio_path: str
    transcript_path: Optional[str] = None
    transcript_text: Optional[str] = None
    label: Optional[int] = None
    corpus_id: Optional[int] = None
    corpus_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

