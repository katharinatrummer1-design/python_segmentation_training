from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REQUIRED_COLUMNS = {"segment_id", "individual_id", "session_id"}


class CricketSpectrogramDataset(Dataset):
    def __init__(
        self,
        segment_ids: list[str],
        labels: np.ndarray,
        data_dir: Path,
    ) -> None:
        if len(segment_ids) != len(labels):
            raise ValueError("segment_ids and labels must have the same length.")

        self.segment_ids = [str(segment_id) for segment_id in segment_ids]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.data_dir = Path(data_dir)

    def __len__(self) -> int:
        return len(self.segment_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        segment_id = self.segment_ids[index]
        spectrogram_path = (
            self.data_dir / "processed" / "spectrograms" / f"{segment_id}.npy"
        )
        spectrogram = np.load(spectrogram_path)
        if spectrogram.ndim != 2:
            raise ValueError(
                f"Expected 2D spectrogram for {segment_id}, got shape {spectrogram.shape}."
            )

        tensor = torch.from_numpy(np.asarray(spectrogram, dtype=np.float32)).unsqueeze(0)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return tensor, label


def _split_frame(
    spectrogram_index_df: pd.DataFrame,
    split_dict: dict,
    split_name: str,
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(spectrogram_index_df.columns)
    if missing_columns:
        raise ValueError(
            "spectrogram_index_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    segment_ids = [str(segment_id) for segment_id in split_dict.get(split_name, [])]
    working_df = spectrogram_index_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str)
    working_df["individual_id"] = working_df["individual_id"].astype(str)
    working_df["session_id"] = working_df["session_id"].astype(str)

    if not segment_ids:
        return working_df.iloc[0:0].copy()

    indexed_df = (
        working_df.drop_duplicates(subset=["segment_id"], keep="last")
        .set_index("segment_id", drop=False)
        .sort_index()
    )
    missing_segment_ids = [
        segment_id for segment_id in segment_ids if segment_id not in indexed_df.index
    ]
    if missing_segment_ids:
        raise ValueError(
            f"{split_name} references spectrograms that are missing from the index: "
            + ", ".join(sorted(missing_segment_ids))
        )

    return indexed_df.loc[segment_ids].reset_index(drop=True).copy()


def _sample_limit(cnn_config: dict, split_name: str) -> int | None:
    raw_limit = cnn_config.get(f"max_{split_name}_samples_per_class")
    if raw_limit in (None, "", False):
        return None

    limit = int(raw_limit)
    if limit <= 0:
        return None
    return limit


def _limit_split_frame(
    split_df: pd.DataFrame,
    split_name: str,
    config: dict,
) -> pd.DataFrame:
    cnn_config = config.get("cnn", {})
    limit = _sample_limit(cnn_config, split_name)
    if limit is None or split_df.empty:
        return split_df

    seed = int(config.get("seed", 0)) + {"train": 0, "val": 101, "test": 202}[split_name]
    sampled_frames: list[pd.DataFrame] = []
    sorted_df = split_df.sort_values(["individual_id", "segment_id"]).reset_index(drop=True)
    for _, class_df in sorted_df.groupby("individual_id", sort=True):
        if len(class_df) <= limit:
            sampled_frames.append(class_df)
            continue

        sampled_frames.append(
            class_df.sample(n=limit, random_state=seed, replace=False)
            .sort_values("segment_id")
            .reset_index(drop=True)
        )

    if not sampled_frames:
        return split_df.iloc[0:0].copy()

    return (
        pd.concat(sampled_frames, ignore_index=True)
        .sort_values(["individual_id", "segment_id"])
        .reset_index(drop=True)
    )


def create_dataloaders(
    spectrogram_index_df: pd.DataFrame,
    split_dict: dict,
    label_encoder: LabelEncoder,
    data_dir: Path,
    config: dict,
) -> dict[str, DataLoader]:
    cnn_config = config.get("cnn", {})
    batch_size = int(cnn_config.get("batch_size", 32))
    num_workers = int(cnn_config.get("num_workers", 0))
    pin_memory = bool(cnn_config.get("pin_memory", torch.cuda.is_available()))

    dataloaders: dict[str, DataLoader] = {}
    for split_name in ("train", "val", "test"):
        split_df = _limit_split_frame(
            _split_frame(spectrogram_index_df, split_dict, split_name),
            split_name,
            config,
        )
        if split_df.empty:
            labels = np.empty((0,), dtype=np.int64)
        else:
            labels = label_encoder.transform(
                split_df["individual_id"].astype(str).to_numpy()
            ).astype(np.int64, copy=False)

        dataset = CricketSpectrogramDataset(
            split_df["segment_id"].astype(str).tolist(),
            labels,
            data_dir,
        )

        use_balanced_sampler = (
            split_name == "train"
            and len(dataset) > 0
            and bool(cnn_config.get("balanced_sampler", False))
        )
        sampler = None
        shuffle = split_name == "train" and len(dataset) > 0
        if use_balanced_sampler:
            class_counts = np.bincount(labels, minlength=int(labels.max()) + 1)
            per_sample_weight = np.asarray(
                [1.0 / max(int(class_counts[label]), 1) for label in labels],
                dtype=np.float64,
            )
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(per_sample_weight, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
            )
            shuffle = False  # sampler and shuffle are mutually exclusive

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return dataloaders
