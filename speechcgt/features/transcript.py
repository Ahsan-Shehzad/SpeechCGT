from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..constants import (
    CLAUSE_LINKING_DEPS,
    CONTENT_POS_TAGS,
    COOKIE_THEFT_INFORMATION_UNITS,
    FILLER_TOKENS,
    PRONOUN_POS_TAGS,
    SELF_REPAIR_MARKERS,
    SEQUENCING_CONNECTIVES,
)
from ..data.schemas import PreprocessedSample

TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")
PRONOUN_FALLBACK = {"he", "she", "it", "they", "them", "his", "her", "their", "him", "we", "i", "you"}

_SPACY_CACHE: dict[str, Any] = {}


@dataclass
class ParsedSentence:
    text: str
    lemmas: list[str]
    pos_tags: list[str]
    dep_labels: list[str]
    head_indices: list[int]


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


def _regex_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _parse_utterances(utterances: list[str], transcript_config: dict[str, Any]) -> list[ParsedSentence]:
    nlp = _get_spacy_pipeline(transcript_config.get("spacy_model", "en_core_web_sm"))
    if nlp is None:
        return [
            ParsedSentence(
                text=utterance,
                lemmas=_regex_tokens(utterance),
                pos_tags=[""] * len(_regex_tokens(utterance)),
                dep_labels=[""] * len(_regex_tokens(utterance)),
                head_indices=list(range(len(_regex_tokens(utterance)))),
            )
            for utterance in utterances
        ]
    parsed: list[ParsedSentence] = []
    for utterance in utterances:
        doc = nlp(utterance)
        tokens = [token for token in doc if token.is_alpha]
        filtered_index = {token.i: idx for idx, token in enumerate(tokens)}
        parsed.append(
            ParsedSentence(
                text=utterance,
                lemmas=[(token.lemma_ or token.text).lower() for token in tokens],
                pos_tags=[token.pos_ for token in tokens],
                dep_labels=[token.dep_ for token in tokens],
                head_indices=[filtered_index.get(token.head.i, idx) for idx, token in enumerate(tokens)],
            )
        )
    return parsed


def _tree_depth(dep_labels: list[str], head_indices: list[int]) -> Optional[float]:
    if not dep_labels or not any(dep_labels):
        return None
    depths = []
    for index in range(len(dep_labels)):
        depth = 0
        current = index
        visited = set()
        while 0 <= current < len(head_indices) and current not in visited:
            visited.add(current)
            next_idx = head_indices[current]
            if next_idx == current:
                break
            current = next_idx
            depth += 1
            if depth > len(head_indices):
                break
        depths.append(depth)
    return float(max(depths)) if depths else None


def _moving_average_ttr(tokens: list[str], window: int) -> Optional[float]:
    if not tokens:
        return None
    if len(tokens) < window:
        return float(len(set(tokens)) / max(len(tokens), 1))
    values = []
    for start in range(0, len(tokens) - window + 1):
        segment = tokens[start : start + window]
        values.append(len(set(segment)) / window)
    return float(np.mean(values)) if values else None


def extract_transcript_features(
    sample: PreprocessedSample,
    transcript_config: dict[str, Any] | None = None,
) -> dict[str, Optional[float]]:
    transcript_config = transcript_config or {}
    utterances = sample.utterances or ([sample.cleaned_transcript] if sample.cleaned_transcript else [])
    parsed = _parse_utterances(utterances, transcript_config)
    alpha_tokens = [token for sentence in parsed for token in sentence.lemmas if token]
    raw_tokens = _regex_tokens(sample.cleaned_transcript)
    total_tokens = len(raw_tokens)
    total_alpha = len(alpha_tokens)
    if total_alpha == 0:
        return {name: None for name in [
            "filled_pause_rate",
            "self_repair_rate",
            "utterance_length",
            "lexical_richness",
            "content_word_density",
            "pronoun_dependency",
            "syntactic_depth",
            "clause_linking_ratio",
            "semantic_coherence",
            "information_unit_density",
            "discourse_sequencing",
        ]}
    filler_count = sum(token in FILLER_TOKENS for token in raw_tokens)
    repair_count = sum(sample.cleaned_transcript.count(marker) for marker in SELF_REPAIR_MARKERS)
    utterance_token_counts = [len(sentence.lemmas) for sentence in parsed if sentence.lemmas]
    content_count = 0
    pronoun_count = 0
    clause_count = 0
    depths = []
    for sentence in parsed:
        if sentence.pos_tags and any(sentence.pos_tags):
            content_count += sum(tag in CONTENT_POS_TAGS for tag in sentence.pos_tags)
            pronoun_count += sum(tag in PRONOUN_POS_TAGS for tag in sentence.pos_tags)
        else:
            content_count += sum(token not in PRONOUN_FALLBACK and len(token) > 2 for token in sentence.lemmas)
            pronoun_count += sum(token in PRONOUN_FALLBACK for token in sentence.lemmas)
        if sentence.dep_labels and any(sentence.dep_labels):
            clause_count += sum(dep in CLAUSE_LINKING_DEPS for dep in sentence.dep_labels)
            depth = _tree_depth(sentence.dep_labels, sentence.head_indices)
            if depth is not None:
                depths.append(depth)
        else:
            clause_count += sum(token in {"and", "but", "because", "while", "when", "that"} for token in sentence.lemmas)
    coherence = None
    if len(utterances) >= 2:
        try:
            vectorizer = TfidfVectorizer()
            matrix = vectorizer.fit_transform(utterances)
            sims = []
            for index in range(matrix.shape[0] - 1):
                current = matrix[index]
                nxt = matrix[index + 1]
                numerator = float(current.multiply(nxt).sum())
                denominator = math.sqrt(float(current.multiply(current).sum()) * float(nxt.multiply(nxt).sum()))
                sims.append(numerator / denominator if denominator > 0 else 0.0)
            coherence = float(np.mean(sims)) if sims else None
        except Exception:
            coherence = None
    info_units = sum(token in COOKIE_THEFT_INFORMATION_UNITS for token in alpha_tokens)
    sequencing = sum(token in SEQUENCING_CONNECTIVES for token in alpha_tokens)
    return {
        "filled_pause_rate": float(filler_count / total_alpha),
        "self_repair_rate": float((repair_count * 100.0) / total_alpha),
        "utterance_length": float(np.mean(utterance_token_counts)) if utterance_token_counts else None,
        "lexical_richness": _moving_average_ttr(alpha_tokens, int(transcript_config.get("mattr_window", 20))),
        "content_word_density": float(content_count / total_alpha),
        "pronoun_dependency": float(pronoun_count / total_alpha),
        "syntactic_depth": float(np.mean(depths)) if depths else None,
        "clause_linking_ratio": float(clause_count / max(len(parsed), 1)),
        "semantic_coherence": coherence,
        "information_unit_density": float(info_units / total_alpha),
        "discourse_sequencing": float(sequencing / max(len(parsed), 1)),
    }
