from __future__ import annotations

from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from .schemas import SpeechSample


class SpeechCGTDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame) -> None:
        self.manifest_df = manifest_df.reset_index(drop=True).copy()

    def __len__(self) -> int:
        return len(self.manifest_df)

    def __getitem__(self, index: int) -> SpeechSample:
        row = self.manifest_df.iloc[index]
        label = row.get("label")
        if pd.isna(label):
            label = None
        else:
            label = int(label)
        return SpeechSample(
            sample_id=str(row["sample_id"]),
            participant_id=str(row["participant_id"]),
            audio_path=str(row["audio_path"]),
            transcript=str(row.get("transcript", "")),
            label=label,
            corpus_id=int(row["corpus_id"]),
            corpus_name=str(row.get("corpus_name", "unknown")),
        )

