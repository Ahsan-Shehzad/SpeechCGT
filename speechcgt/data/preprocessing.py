from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from .schemas import PreprocessedSample, SpeechSample

_SPACY_CACHE: dict[str, Any] = {}


def _get_spacy_pipeline(model_name: str):
    if model_name in _SPACY_CACHE:
        return _SPACY_CACHE[model_name]
    try:
        import spacy

        try:
            nlp = spacy.load(model_name)
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        _SPACY_CACHE[model_name] = nlp
        return nlp
    except Exception:
        return None


def _clean_transcript_text(text: str, transcript_config: dict[str, Any]) -> str:
    cleaned = text or ""
    if transcript_config.get("remove_chat_control_symbols", True):
        cleaned = re.sub(r"\x15.*?\x15", " ", cleaned)
        cleaned = re.sub(r"\[[^\]]*?\]", " ", cleaned)
        cleaned = re.sub(r"&=\w+", " ", cleaned)
        cleaned = re.sub(r"@\w+", " ", cleaned)
    cleaned = re.sub(r"[<>]", " ", cleaned)
    cleaned = re.sub(r"[_~]", " ", cleaned)
    if transcript_config.get("normalize_punctuation", True):
        cleaned = cleaned.replace(";", ",").replace(":", ": ")
        cleaned = re.sub(r"[“”]", "\"", cleaned)
        cleaned = re.sub(r"[‘’]", "'", cleaned)
    if transcript_config.get("lowercase", True):
        cleaned = cleaned.lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _segment_utterances(original_text: str, cleaned_text: str, transcript_config: dict[str, Any]) -> list[str]:
    line_utterances = [line.strip() for line in original_text.splitlines() if line.strip()]
    if len(line_utterances) > 1:
        return [_clean_transcript_text(line, transcript_config) for line in line_utterances if line.strip()]
    nlp = _get_spacy_pipeline(transcript_config.get("spacy_model", "en_core_web_sm"))
    if nlp is not None:
        doc = nlp(cleaned_text)
        utterances = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if utterances:
            return utterances
    utterances = [segment.strip() for segment in re.split(r"[.!?]+", cleaned_text) if segment.strip()]
    return utterances or ([cleaned_text] if cleaned_text else [])


def _load_and_normalize_audio(audio_path: str, audio_config: dict[str, Any]) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(audio_path, always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1 and audio_config.get("mono", True):
        waveform = waveform.mean(axis=1, dtype=np.float32)
    target_sr = int(audio_config.get("target_sample_rate", 16000))
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    if np.max(np.abs(waveform)) > 0:
        try:
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(waveform)
            waveform = pyln.normalize.loudness(waveform, loudness, audio_config.get("target_lufs", -23.0))
        except Exception:
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak
    return waveform.astype(np.float32), sample_rate


def preprocess_sample(
    sample: SpeechSample,
    target_sample_rate: int,
    target_lufs: float,
    transcript_config: dict[str, Any] | None = None,
    audio_config: dict[str, Any] | None = None,
) -> PreprocessedSample:
    transcript_config = transcript_config or {}
    audio_config = dict(audio_config or {})
    audio_config.setdefault("target_sample_rate", target_sample_rate)
    audio_config.setdefault("target_lufs", target_lufs)
    waveform, sample_rate = _load_and_normalize_audio(sample.audio_path, audio_config)
    original_transcript = sample.transcript or ""
    cleaned_transcript = _clean_transcript_text(original_transcript, transcript_config)
    utterances = _segment_utterances(original_transcript, cleaned_transcript, transcript_config)
    return PreprocessedSample(
        sample=sample,
        waveform=waveform,
        sample_rate=sample_rate,
        cleaned_transcript=cleaned_transcript,
        utterances=utterances,
    )

