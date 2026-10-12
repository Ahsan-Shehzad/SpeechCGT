from __future__ import annotations

from typing import Any, Optional

import librosa
import numpy as np

from ..data.schemas import PreprocessedSample


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _semitones(f0_hz: np.ndarray, reference_hz: float = 55.0) -> np.ndarray:
    return 12.0 * np.log2(np.clip(f0_hz, 1e-6, None) / reference_hz)


def _extract_pitch(
    waveform: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    f0, voiced_flag, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced = np.asarray(voiced_flag, dtype=bool)
    return f0, voiced


def _rms_db(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length).squeeze(0)
    return librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)


def _pause_regions(pause_mask: np.ndarray, hop_length: int, sample_rate: int) -> list[tuple[int, int, float]]:
    regions: list[tuple[int, int, float]] = []
    start = None
    for index, is_pause in enumerate(pause_mask.tolist()):
        if is_pause and start is None:
            start = index
        elif not is_pause and start is not None:
            duration = (index - start) * hop_length / sample_rate
            regions.append((start, index, duration))
            start = None
    if start is not None:
        duration = (len(pause_mask) - start) * hop_length / sample_rate
        regions.append((start, len(pause_mask), duration))
    return regions


def _syllable_proxy(waveform: np.ndarray, sample_rate: int, hop_length: int) -> float:
    onset_envelope = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=hop_length)
    peaks = librosa.util.peak_pick(onset_envelope, pre_max=2, post_max=2, pre_avg=3, post_avg=3, delta=0.15, wait=2)
    return float(len(peaks))


def _prosodic_reset(
    semitone_f0: np.ndarray,
    voiced: np.ndarray,
    pause_regions: list[tuple[int, int, float]],
    min_pause_seconds: float,
) -> Optional[float]:
    voiced_segments: list[tuple[int, int]] = []
    start = None
    for idx, is_voiced in enumerate(voiced.tolist()):
        if is_voiced and start is None:
            start = idx
        elif not is_voiced and start is not None:
            voiced_segments.append((start, idx))
            start = None
    if start is not None:
        voiced_segments.append((start, len(voiced)))
    if len(voiced_segments) < 2:
        return None
    resets: list[float] = []
    for current_index in range(len(voiced_segments) - 1):
        left = voiced_segments[current_index]
        right = voiced_segments[current_index + 1]
        qualifying_pause = [
            duration
            for start, end, duration in pause_regions
            if start >= left[1] and end <= right[0] and duration >= min_pause_seconds
        ]
        if not qualifying_pause:
            continue
        left_values = semitone_f0[left[0] : left[1]]
        right_values = semitone_f0[right[0] : right[1]]
        if left_values.size == 0 or right_values.size == 0:
            continue
        resets.append(float(abs(np.median(left_values) - np.median(right_values))))
    return float(np.mean(resets)) if resets else None


def _jitter_shimmer(
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    rms_db: np.ndarray,
) -> Optional[float]:
    voiced_f0 = f0_hz[voiced & (f0_hz > 0)]
    if voiced_f0.size < 3:
        return None
    periods = 1.0 / np.clip(voiced_f0, 1e-6, None)
    jitter = np.mean(np.abs(np.diff(periods)) / np.clip(periods[:-1], 1e-6, None))
    voiced_rms = rms_db[: len(voiced)][voiced[: len(rms_db)]]
    if voiced_rms.size < 3:
        return float(np.log1p(jitter))
    linear_amp = np.exp(voiced_rms / 20.0)
    shimmer = np.mean(np.abs(np.diff(linear_amp)) / np.clip(linear_amp[:-1], 1e-6, None))
    return float(np.mean([np.log1p(jitter), np.log1p(shimmer)]))


def _harmonics_to_noise_ratio(waveform: np.ndarray) -> Optional[float]:
    if waveform.size == 0:
        return None
    harmonic, _ = librosa.effects.hpss(waveform)
    noise = waveform - harmonic
    harmonic_energy = float(np.mean(harmonic**2))
    noise_energy = float(np.mean(noise**2))
    return float(10.0 * np.log10((harmonic_energy + 1e-8) / (noise_energy + 1e-8)))


def _cepstral_peak_prominence(
    waveform: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
) -> Optional[float]:
    if waveform.size < frame_length:
        return None
    values: list[float] = []
    for start in range(0, len(waveform) - frame_length, hop_length):
        frame = waveform[start : start + frame_length]
        windowed = frame * np.hanning(frame_length)
        spectrum = np.fft.rfft(windowed)
        log_mag = np.log(np.abs(spectrum) + 1e-8)
        cepstrum = np.fft.irfft(log_mag)
        q_min = int(sample_rate / 500)
        q_max = int(sample_rate / 60)
        if q_max <= q_min or q_max >= len(cepstrum):
            continue
        region = cepstrum[q_min:q_max]
        peak = float(np.max(region))
        baseline = float(np.median(region))
        values.append(peak - baseline)
    return float(np.mean(values)) if values else None


def _formant_stability(
    waveform: np.ndarray,
    sample_rate: int,
) -> Optional[float]:
    try:
        import parselmouth
    except Exception:
        return None
    try:
        sound = parselmouth.Sound(waveform, sample_rate)
        formant = sound.to_formant_burg()
        duration = sound.get_total_duration()
        times = np.arange(0.02, duration, 0.01)
        f1_values = []
        f2_values = []
        for time in times:
            f1 = formant.get_value_at_time(1, float(time))
            f2 = formant.get_value_at_time(2, float(time))
            if f1 and np.isfinite(f1):
                f1_values.append(float(f1))
            if f2 and np.isfinite(f2):
                f2_values.append(float(f2))
        if len(f1_values) < 3 or len(f2_values) < 3:
            return None
        f1_cv = np.std(f1_values) / max(abs(np.mean(f1_values)), 1e-6)
        f2_cv = np.std(f2_values) / max(abs(np.mean(f2_values)), 1e-6)
        return float(np.mean([f1_cv, f2_cv]))
    except Exception:
        return None


def extract_acoustic_features(sample: PreprocessedSample, audio_config: dict[str, Any] | None = None) -> dict[str, Optional[float]]:
    audio_config = audio_config or {}
    waveform = sample.waveform.astype(np.float32)
    sample_rate = sample.sample_rate
    frame_length = int(audio_config.get("frame_length", 512))
    hop_length = int(audio_config.get("hop_length", 160))
    pause_threshold_db = float(audio_config.get("pause_threshold_db", -40.0))
    min_pause_seconds = float(audio_config.get("min_pause_seconds", 0.2))
    total_duration = len(waveform) / max(sample_rate, 1)
    if total_duration <= 0:
        return {name: None for name in [
            "pitch_variability",
            "pitch_range",
            "loudness_variability",
            "speech_rate",
            "articulation_rate",
            "prosodic_reset",
            "jitter_shimmer",
            "harmonics_to_noise",
            "cepstral_prominence",
            "formant_stability",
            "silence_ratio",
            "mean_pause_duration",
            "pause_frequency",
        ]}
    rms_db = _rms_db(waveform, frame_length, hop_length)
    pause_mask = rms_db < pause_threshold_db
    pause_regions = _pause_regions(pause_mask, hop_length, sample_rate)
    paused_duration = sum(duration for _, _, duration in pause_regions if duration >= min_pause_seconds)
    valid_pause_durations = [duration for _, _, duration in pause_regions if duration >= min_pause_seconds]
    syllables = _syllable_proxy(waveform, sample_rate, hop_length)
    try:
        f0_hz, voiced = _extract_pitch(waveform, sample_rate, frame_length, hop_length)
    except Exception:
        f0_hz = np.zeros(len(rms_db), dtype=np.float32)
        voiced = np.zeros(len(rms_db), dtype=bool)
    voiced_f0 = f0_hz[voiced & (f0_hz > 0)]
    voiced_semitones = _semitones(voiced_f0) if voiced_f0.size else np.array([], dtype=np.float32)
    prosodic_reset = _prosodic_reset(
        _semitones(np.clip(f0_hz, 1e-6, None)),
        voiced,
        pause_regions,
        min_pause_seconds,
    )
    speech_rate = _safe_ratio(syllables, total_duration)
    articulation_rate = _safe_ratio(syllables, max(total_duration - paused_duration, 1e-6))
    pause_frequency = _safe_ratio(len(valid_pause_durations) * 60.0, total_duration)
    return {
        "pitch_variability": float(np.std(voiced_semitones)) if voiced_semitones.size > 1 else None,
        "pitch_range": float(np.percentile(voiced_semitones, 95) - np.percentile(voiced_semitones, 5)) if voiced_semitones.size > 1 else None,
        "loudness_variability": float(np.std(rms_db)) if rms_db.size > 1 else None,
        "speech_rate": speech_rate,
        "articulation_rate": articulation_rate,
        "prosodic_reset": prosodic_reset,
        "jitter_shimmer": _jitter_shimmer(f0_hz, voiced, rms_db),
        "harmonics_to_noise": _harmonics_to_noise_ratio(waveform),
        "cepstral_prominence": _cepstral_peak_prominence(waveform, sample_rate, frame_length, hop_length),
        "formant_stability": _formant_stability(waveform, sample_rate),
        "silence_ratio": _safe_ratio(paused_duration, total_duration),
        "mean_pause_duration": float(np.mean(valid_pause_durations)) if valid_pause_durations else None,
        "pause_frequency": pause_frequency,
    }
