from __future__ import annotations

from .schemas import SpeechSample


def collate_speech_samples(batch: list[SpeechSample]) -> list[SpeechSample]:
    return batch

