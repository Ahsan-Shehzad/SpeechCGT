from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from ..constants import CORPUS_IDS
from ..utils.io import save_json
from .schemas import SplitManifest


def load_corpus_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_path = Path(manifest_path)
    if manifest_path.suffix.lower() == ".csv":
        return pd.read_csv(manifest_path)
    if manifest_path.suffix.lower() == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            return pd.DataFrame(json.load(handle))
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _participant_frame(manifest_df: pd.DataFrame, participant_id_key: str, label_key: str) -> pd.DataFrame:
    participant_df = (
        manifest_df[[participant_id_key, label_key]]
        .dropna(subset=[participant_id_key, label_key])
        .drop_duplicates(subset=[participant_id_key])
        .reset_index(drop=True)
    )
    return participant_df


def build_group_stratified_splits(
    manifest_df: pd.DataFrame,
    num_folds: int,
    group_key: str,
    label_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    source_corpus: str | None = None,
) -> list[SplitManifest]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.")
    working_df = manifest_df.copy()
    if source_corpus is not None:
        source_id = CORPUS_IDS[source_corpus]
        working_df = working_df[working_df["corpus_id"] == source_id].reset_index(drop=True)
    participant_df = _participant_frame(working_df, group_key, label_key)
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds: list[SplitManifest] = []
    val_fraction_within_train_val = val_ratio / (train_ratio + val_ratio)
    for fold_id, (train_val_idx, test_idx) in enumerate(
        splitter.split(working_df, working_df[label_key], working_df[group_key])
    ):
        test_df = working_df.iloc[test_idx]
        train_val_df = working_df.iloc[train_val_idx]
        participant_train_val = _participant_frame(train_val_df, group_key, label_key)
        train_participants, val_participants = train_test_split(
            participant_train_val[group_key],
            test_size=val_fraction_within_train_val,
            stratify=participant_train_val[label_key],
            random_state=seed + fold_id,
        )
        train_participants = sorted(train_participants.tolist())
        val_participants = sorted(val_participants.tolist())
        test_participants = sorted(test_df[group_key].drop_duplicates().astype(str).tolist())
        train_df = train_val_df[train_val_df[group_key].astype(str).isin(train_participants)]
        val_df = train_val_df[train_val_df[group_key].astype(str).isin(val_participants)]
        external_mask = manifest_df["corpus_id"] == CORPUS_IDS["WLS"]
        external_df = manifest_df[external_mask]
        folds.append(
            SplitManifest(
                source_corpus=source_corpus or "unknown",
                fold_id=fold_id,
                train_sample_ids=sorted(train_df["sample_id"].astype(str).tolist()),
                val_sample_ids=sorted(val_df["sample_id"].astype(str).tolist()),
                test_sample_ids=sorted(test_df["sample_id"].astype(str).tolist()),
                external_sample_ids=sorted(external_df["sample_id"].astype(str).tolist()),
                train_participant_ids=train_participants,
                val_participant_ids=val_participants,
                test_participant_ids=test_participants,
                external_participant_ids=sorted(external_df[group_key].astype(str).drop_duplicates().tolist()),
            )
        )
    return folds


def save_split_manifest(manifest: SplitManifest, output_path: str | Path) -> None:
    save_json(manifest.to_dict(), output_path)

