from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd


SPLIT_NAMES = ("train", "val", "test")
REQUIRED_COLUMNS = {"segment_id", "individual_id", "session_id", "recording_id"}


def _normalized_ratios(config: dict) -> tuple[float, float, float]:
    split_config = config.get("splits", {})
    ratios = (
        float(split_config.get("train_ratio", 0.7)),
        float(split_config.get("val_ratio", 0.15)),
        float(split_config.get("test_ratio", 0.15)),
    )
    total = sum(ratios)
    if total <= 0.0:
        raise ValueError("Split ratios must sum to a positive value.")
    return tuple(ratio / total for ratio in ratios)


def _allocate_split_counts(
    session_count: int,
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> tuple[int, int, int]:
    if session_count <= 0:
        return (0, 0, 0)

    # Largest-remainder (Hamilton) apportionment: floor each ideal count, then
    # hand the leftover whole sessions to the splits with the biggest fractional
    # remainders. Guarantees the integer counts sum exactly to session_count while
    # staying as close as possible to the requested train/val/test ratios.
    raw_counts = [session_count * ratio for ratio in ratios]
    counts = [math.floor(count) for count in raw_counts]
    remainder = session_count - sum(counts)

    if remainder > 0:
        ordering = sorted(
            range(len(raw_counts)),
            key=lambda index: (
                raw_counts[index] - counts[index],
                rng.random(),
            ),
            reverse=True,
        )
        for index in ordering[:remainder]:
            counts[index] += 1

    # Invariant: every individual MUST appear in train. If rounding left train
    # empty, steal one session from the largest other split.
    if counts[0] == 0:
        donor_candidates = [
            index for index in range(1, len(counts)) if counts[index] > 0
        ] or [index for index, count in enumerate(counts) if count > 0]
        if not donor_candidates:
            counts[0] = 1
        else:
            donor = max(
                donor_candidates,
                key=lambda index: (counts[index], raw_counts[index], rng.random()),
            )
            counts[donor] -= 1
            counts[0] += 1

    if sum(counts) != session_count:
        raise AssertionError("Allocated split counts do not match session count.")

    return tuple(int(count) for count in counts)


def _session_sort_key(value: object) -> str:
    return str(value)


def _group_column(config: dict) -> str:
    group_by = str(config.get("splits", {}).get("group_by", "session_id"))
    if group_by not in {"session_id", "recording_id"}:
        raise ValueError(
            "splits.group_by must be 'session_id' or 'recording_id', "
            f"got: {group_by!r}"
        )
    return group_by


def build_session_splits(segments_df: pd.DataFrame, config: dict) -> dict:
    missing_columns = REQUIRED_COLUMNS.difference(segments_df.columns)
    if missing_columns:
        raise ValueError(
            "segments_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    group_col = _group_column(config)

    group_owner_counts = (
        segments_df.groupby(group_col, dropna=False)["individual_id"].nunique()
    )
    conflicting_groups = group_owner_counts.loc[group_owner_counts > 1]
    if not conflicting_groups.empty:
        raise ValueError(
            f"Each {group_col} must map to exactly one individual_id. "
            f"Conflicts found for: {', '.join(map(str, conflicting_groups.index.tolist()))}"
        )

    seed = int(config.get("seed", 0))
    ratios = _normalized_ratios(config)
    split_segments: dict[str, list[str]] = {split_name: [] for split_name in SPLIT_NAMES}
    session_map: dict[str, str] = {}
    metadata_individuals: dict[str, dict[str, object]] = {}

    grouped = segments_df.groupby("individual_id", dropna=False, sort=True)
    for individual_index, (individual_id, individual_segments_df) in enumerate(grouped):
        rng = random.Random(seed + individual_index)
        session_ids = sorted(
            individual_segments_df[group_col].drop_duplicates().tolist(),
            key=_session_sort_key,
        )
        shuffled_session_ids = session_ids.copy()
        rng.shuffle(shuffled_session_ids)
        train_count, val_count, test_count = _allocate_split_counts(
            len(shuffled_session_ids),
            ratios,
            rng,
        )

        split_boundaries = {
            "train": shuffled_session_ids[:train_count],
            "val": shuffled_session_ids[train_count : train_count + val_count],
            "test": shuffled_session_ids[
                train_count + val_count : train_count + val_count + test_count
            ],
        }

        for split_name, session_values in split_boundaries.items():
            for session_id in sorted(session_values, key=_session_sort_key):
                session_key = str(session_id)
                session_map[session_key] = split_name

        metadata_individuals[str(individual_id)] = {
            "session_count": len(session_ids),
            "train_sessions": sorted(
                [str(session_id) for session_id in split_boundaries["train"]]
            ),
            "val_sessions": sorted(
                [str(session_id) for session_id in split_boundaries["val"]]
            ),
            "test_sessions": sorted(
                [str(session_id) for session_id in split_boundaries["test"]]
            ),
        }

    split_assignment = segments_df[group_col].map(lambda group_id: session_map.get(str(group_id)))
    if split_assignment.isna().any():
        missing_sessions = (
            segments_df.loc[split_assignment.isna(), group_col].astype(str).unique().tolist()
        )
        raise ValueError(
            f"Some {group_col} groups were not assigned to a split: "
            + ", ".join(sorted(missing_sessions))
        )

    for split_name in SPLIT_NAMES:
        segment_ids = (
            segments_df.loc[split_assignment == split_name, "segment_id"]
            .astype(str)
            .tolist()
        )
        split_segments[split_name] = segment_ids

    split_metadata = {
        "seed": seed,
        "group_by": group_col,
        "ratios": {
            "train": ratios[0],
            "val": ratios[1],
            "test": ratios[2],
        },
        "counts": {
            "segments": {
                split_name: len(split_segments[split_name]) for split_name in SPLIT_NAMES
            },
            "sessions": {
                split_name: sum(
                    1 for assigned_split in session_map.values() if assigned_split == split_name
                )
                for split_name in SPLIT_NAMES
            },
            "individuals": int(segments_df["individual_id"].nunique()),
        },
        "per_individual": metadata_individuals,
    }

    return {
        **split_segments,
        "session_map": dict(sorted(session_map.items())),
        "metadata": split_metadata,
    }


def validate_split_integrity(split_dict: dict, segments_df: pd.DataFrame) -> list[str]:
    missing_columns = REQUIRED_COLUMNS.difference(segments_df.columns)
    if missing_columns:
        raise ValueError(
            "segments_df is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    issues: list[str] = []
    working_df = segments_df.copy()
    working_df["segment_id"] = working_df["segment_id"].astype(str)
    working_df["session_id"] = working_df["session_id"].astype(str)
    working_df["recording_id"] = working_df["recording_id"].astype(str)
    working_df["individual_id"] = working_df["individual_id"].astype(str)

    split_segment_sets = {
        split_name: set(map(str, split_dict.get(split_name, []))) for split_name in SPLIT_NAMES
    }

    all_assigned_segment_ids: set[str] = set()
    for split_name, segment_ids in split_segment_sets.items():
        duplicate_segments = all_assigned_segment_ids.intersection(segment_ids)
        if duplicate_segments:
            issues.append(
                f"Segment overlap detected for {split_name}: "
                + ", ".join(sorted(duplicate_segments))
            )
        all_assigned_segment_ids.update(segment_ids)

    all_segment_ids = set(working_df["segment_id"].tolist())
    unassigned_segment_ids = all_segment_ids.difference(all_assigned_segment_ids)
    if unassigned_segment_ids:
        issues.append(
            "Unassigned segments detected: " + ", ".join(sorted(unassigned_segment_ids))
        )

    unknown_segment_ids = all_assigned_segment_ids.difference(all_segment_ids)
    if unknown_segment_ids:
        issues.append(
            "Unknown segment_ids referenced in split_dict: "
            + ", ".join(sorted(unknown_segment_ids))
        )

    split_session_sets: dict[str, set[str]] = {}
    split_recording_sets: dict[str, set[str]] = {}
    split_individual_sets: dict[str, set[str]] = {}
    for split_name, segment_ids in split_segment_sets.items():
        split_rows_df = working_df.loc[working_df["segment_id"].isin(segment_ids)]
        split_session_sets[split_name] = set(split_rows_df["session_id"].tolist())
        split_recording_sets[split_name] = set(split_rows_df["recording_id"].tolist())
        split_individual_sets[split_name] = set(split_rows_df["individual_id"].tolist())

    # When grouping by recording_id, segments from one WAV always stay together
    # (the leakage guarantee per FEATURES_BEREINIGUNG_GRILLEN.md), but a single
    # session (e.g. a recording day) can legitimately span splits, so the
    # session-overlap check only applies when grouping by session_id.
    group_by = str(split_dict.get("metadata", {}).get("group_by", "session_id"))
    check_session_overlap = group_by == "session_id"

    for index, left_split in enumerate(SPLIT_NAMES):
        for right_split in SPLIT_NAMES[index + 1 :]:
            if check_session_overlap:
                shared_sessions = split_session_sets[left_split].intersection(
                    split_session_sets[right_split]
                )
                if shared_sessions:
                    issues.append(
                        f"Session overlap between {left_split} and {right_split}: "
                        + ", ".join(sorted(shared_sessions))
                    )

            shared_recordings = split_recording_sets[left_split].intersection(
                split_recording_sets[right_split]
            )
            if shared_recordings:
                issues.append(
                    f"Recording overlap between {left_split} and {right_split}: "
                    + ", ".join(sorted(shared_recordings))
                )

    all_individuals = set(working_df["individual_id"].tolist())
    train_individuals = split_individual_sets["train"]
    missing_train_individuals = all_individuals.difference(train_individuals)
    if missing_train_individuals:
        issues.append(
            "Individuals missing from train split: "
            + ", ".join(sorted(missing_train_individuals))
        )

    for split_name in ("val", "test"):
        out_of_train_individuals = split_individual_sets[split_name].difference(train_individuals)
        if out_of_train_individuals:
            issues.append(
                f"{split_name} contains individuals absent from train: "
                + ", ".join(sorted(out_of_train_individuals))
            )

    return issues


def save_splits(split_dict: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(split_dict, handle, indent=2, sort_keys=False)
        handle.write("\n")
