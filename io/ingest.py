"""Ingestion / Bereinigung for the Katharina *Gryllus campestris* dataset.

This module turns the raw folder tree under ``katharina grillen/merged`` into a
clean, deterministic ``recordings.csv`` manifest that the rest of the
``cricket_id`` pipeline can consume, following the rules documented in
``katharina grillen/FEATURES_BEREINIGUNG_GRILLEN.md``:

* recursive ``.wav`` scan (case-insensitive),
* filename + path validation against the calling / courtship conventions,
* normalization of known structural issues (``courthsip`` typo, ``G_campestris_3``
  vs ``G_campestris_03``, session sub-folders),
* deduplication of download artefacts (`` (1)``, `` (2)`` ...) by content hash,
* technical audio QC (sample rate, duration, dBFS, clipping ratio),
* a stable mapping into the program's recording-manifest schema.

Raw files are never modified; everything is written to ``data/raw`` /
``data/interim``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


# Folder name (relative to the merged root) -> (context, song_type, daytime)
GROUP_MAP: dict[str, tuple[str, str, str]] = {
    "nachts calling": ("night_calling", "calling", "night"),
    "tags calling": ("day_calling", "calling", "day"),
    "tags courtship": ("day_courtship", "courtship", "day"),
}

# Manifest columns required by cricket_id.io.manifest.REQUIRED_RECORDING_COLUMNS
# plus the richer Bereinigung columns kept for diagnostics / future contexts.
MANIFEST_COLUMNS = [
    "recording_id",
    "individual_id",
    "session_id",
    "song_type",
    "context",
    "audio_path",
    "temperature_log_path",
    "recording_start_utc",
    "duration_s",
    "sr",
    "bit_depth",
    "notes",
    # --- extended Bereinigung columns ---
    "group",
    "daytime",
    "female_context",
    "sequence_id",
    "date_iso",
    "time_hhmm",
    "canonical_filename",
    "raw_filename",
    "sha256",
    "file_size_bytes",
    "channels",
    "rms_dbfs",
    "peak_dbfs",
    "clipping_ratio",
    "duplicate_status",
    "qc_status",
    "qc_notes",
]

TEMPERATURE_COLUMNS = ["timestamp_utc", "temperature_c", "recording_id"]

_COPY_SUFFIX_RE = re.compile(r"\s*\((\d+)\)\s*$")
# Filenames vary widely (``Gcampestris_3_2_001_...``, ``G_campestris_02_003_...``,
# ``..._court_014_...``, even ``rec_001_...``) but they all end in
# ``_<DDMMYY>_<HHMM>``. Anchor on that trailing timestamp and take the individual
# ID from the folder, which is the reliable source of identity.
_TIMESTAMP_RE = re.compile(r"_(\d{6})_(\d{4})$")
_INT_TOKEN_RE = re.compile(r"(\d+)")
_NAME_ID_RE = re.compile(r"campestris[_-]?(\d+)", re.IGNORECASE)
_INDIVIDUAL_DIGITS_RE = re.compile(r"(\d+)")


@dataclass
class IngestSummary:
    """Counters and audit notes collected while building the manifest."""

    total_wav_files: int = 0
    kept: int = 0
    exact_duplicates: int = 0
    name_duplicates: int = 0
    unparsed: int = 0
    qc_warn: int = 0
    qc_fail: int = 0
    normalizations: list[str] = field(default_factory=list)
    skipped_non_audio: list[str] = field(default_factory=list)


def _bit_depth_from_subtype(subtype: str | None) -> int:
    if not subtype:
        return 0
    match = re.search(r"(\d+)", subtype)
    return int(match.group(1)) if match else 0


def _normalize_individual(token: str) -> str:
    """Map ``G_campestris_3`` / ``Gcampestris_03`` -> canonical ``G_campestris_03``."""
    match = _INDIVIDUAL_DIGITS_RE.search(token)
    if not match:
        return token
    return f"G_campestris_{int(match.group(1)):02d}"


def _date_to_iso(ddmmyy: str) -> str:
    day = int(ddmmyy[0:2])
    month = int(ddmmyy[2:4])
    year = 2000 + int(ddmmyy[4:6])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _canonical_stem(stem: str) -> str:
    """Strip a trailing ``(n)`` download/copy suffix from a file stem."""
    return _COPY_SUFFIX_RE.sub("", stem).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")


def _audio_qc(path: Path) -> dict[str, object]:
    """Read the file once and compute the technical QC metrics."""
    info = sf.info(str(path))
    audio, _ = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    n = int(audio.size)
    peak = float(np.max(np.abs(audio))) if n else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if n else 0.0
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.999)) if n else 0.0
    eps = 1e-12
    return {
        "sr": int(info.samplerate),
        "channels": int(info.channels),
        "duration_s": float(info.duration),
        "bit_depth": _bit_depth_from_subtype(info.subtype),
        "rms_dbfs": float(20.0 * np.log10(max(rms, eps))),
        "peak_dbfs": float(20.0 * np.log10(max(peak, eps))),
        "clipping_ratio": clipping_ratio,
        "n_samples": n,
    }


def _qc_status(qc: dict[str, object], thresholds: dict) -> tuple[str, list[str]]:
    notes: list[str] = []
    status = "pass"

    min_duration = float(thresholds.get("min_duration_s", 0.05))
    max_clipping = float(thresholds.get("max_clipping_ratio", 0.01))
    min_rms_dbfs = float(thresholds.get("min_rms_dbfs", -60.0))

    if float(qc["duration_s"]) < min_duration:
        notes.append(f"duration<{min_duration}s")
        status = "fail"
    if float(qc["clipping_ratio"]) > max_clipping:
        notes.append(f"clipping_ratio>{max_clipping}")
        status = "warn" if status == "pass" else status
    if float(qc["rms_dbfs"]) < min_rms_dbfs:
        notes.append(f"rms_dbfs<{min_rms_dbfs}")
        status = "warn" if status == "pass" else status
    if int(qc.get("n_samples", 0)) == 0:
        notes.append("empty_signal")
        status = "fail"
    return status, notes


def _parse_record(
    path: Path,
    merged_root: Path,
    summary: IngestSummary,
) -> dict[str, object] | None:
    """Parse one WAV file into a manifest row (without QC / dedup fields)."""
    rel = path.relative_to(merged_root)
    parts = list(rel.parts)
    if len(parts) < 2:
        summary.unparsed += 1
        return None

    group = parts[0]
    if group not in GROUP_MAP:
        summary.unparsed += 1
        return None
    context, song_type, daytime = GROUP_MAP[group]

    raw_filename = path.name
    canonical_filename = _canonical_stem(path.stem) + path.suffix.lower()
    notes: list[str] = []

    female_context = "NA"
    sub_parts = parts[1:-1]  # folders between group and file
    if song_type == "courtship":
        # tags courtship/<Weibchen X>/<G_campestris_ID_courtship>/file.wav
        if sub_parts:
            female_raw = sub_parts[0]
            female_context = _safe_id(female_raw)
        individual_folder = sub_parts[1] if len(sub_parts) > 1 else sub_parts[0]
        if "courthsip" in individual_folder.lower():
            notes.append(f"typo_normalized:{individual_folder}->courtship")
            summary.normalizations.append(f"{rel}: courthsip typo normalized")
        session_subfolder = ""
    else:
        individual_folder = sub_parts[0] if sub_parts else ""
        session_subfolder = sub_parts[1] if len(sub_parts) > 1 else ""

    individual_id = _normalize_individual(individual_folder)

    stem = _canonical_stem(path.stem)
    match = _TIMESTAMP_RE.search(stem)

    if match is None:
        summary.unparsed += 1
        return {
            "individual_id": individual_id,
            "context": context,
            "song_type": song_type,
            "group": group,
            "daytime": daytime,
            "female_context": female_context,
            "audio_path": str(path.resolve()),
            "raw_filename": raw_filename,
            "canonical_filename": canonical_filename,
            "sequence_id": "",
            "date_iso": "",
            "time_hhmm": "",
            "recording_start_utc": "",
            "session_id": f"{individual_id}__unparsed",
            "_parse_failed": True,
            "notes": "filename_has_no_parseable_timestamp",
        }

    ddmmyy, hhmm = match.group(1), match.group(2)
    head = stem[: match.start()]

    # Sequence number = last integer token before the timestamp (if any).
    int_tokens = _INT_TOKEN_RE.findall(head)
    sequence_id = int_tokens[-1] if int_tokens else ""

    is_courtship_name = "court" in head.lower()

    # Cross-check folder-derived id against any id embedded in the filename.
    name_id_match = _NAME_ID_RE.search(head)
    if name_id_match:
        name_id = int(name_id_match.group(1))
        folder_int = _INDIVIDUAL_DIGITS_RE.search(individual_folder)
        if folder_int and int(folder_int.group(1)) != name_id:
            notes.append(f"id_mismatch_folder={individual_folder}_name={name_id}")
    if song_type == "courtship" and not is_courtship_name:
        notes.append("missing_court_token_in_courtship_file")
    if song_type != "courtship" and is_courtship_name:
        notes.append("court_token_in_calling_file")

    date_iso = _date_to_iso(ddmmyy)
    time_hhmm = hhmm
    recording_start_utc = f"{date_iso}T{hhmm[0:2]}:{hhmm[2:4]}:00Z"

    if session_subfolder:
        session_id = f"{individual_id}__{_safe_id(session_subfolder)}"
        summary.normalizations.append(
            f"{rel}: session derived from subfolder '{session_subfolder}'"
        )
    elif song_type == "courtship":
        session_id = f"{individual_id}__court__{date_iso}"
    else:
        session_id = f"{individual_id}__{daytime}__{date_iso}"

    return {
        "individual_id": individual_id,
        "context": context,
        "song_type": song_type,
        "group": group,
        "daytime": daytime,
        "female_context": female_context,
        "audio_path": str(path.resolve()),
        "raw_filename": raw_filename,
        "canonical_filename": canonical_filename,
        "sequence_id": sequence_id,
        "date_iso": date_iso,
        "time_hhmm": time_hhmm,
        "recording_start_utc": recording_start_utc,
        "session_id": session_id,
        "_parse_failed": False,
        "notes": "; ".join(notes),
    }


def _make_recording_id(canonical_filename: str, used_ids: dict[str, int]) -> str:
    base = _safe_id(Path(canonical_filename).stem)
    if base not in used_ids:
        used_ids[base] = 0
        return base
    used_ids[base] += 1
    return f"{base}__v{used_ids[base]}"


def build_recordings_manifest(
    source_root: str | Path,
    output_dir: str | Path,
    *,
    qc_thresholds: dict | None = None,
    interim_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, IngestSummary]:
    """Scan ``source_root`` and write ``recordings.csv`` + ``temperature.csv``.

    Parameters
    ----------
    source_root:
        Folder that contains the ``merged`` tree (or the ``merged`` folder
        itself). The function locates the ``nachts calling`` / ``tags calling``
        / ``tags courtship`` groups beneath it.
    output_dir:
        Directory where ``recordings.csv`` and ``temperature.csv`` are written
        (typically ``data/raw``).
    qc_thresholds:
        Optional QC thresholds; see ``_qc_status``.
    interim_dir:
        Optional directory for ``duplicate_report.csv`` / ``qc_report.csv``
        (typically ``data/interim/manifest``).
    """
    qc_thresholds = qc_thresholds or {}
    source_path = Path(source_root)
    merged_root = source_path if source_path.name == "merged" else source_path / "merged"
    if not merged_root.exists():
        # Fall back to the source root itself (e.g. a custom layout / fixtures).
        merged_root = source_path
    if not merged_root.exists():
        raise FileNotFoundError(f"Ingestion source not found: {merged_root}")

    summary = IngestSummary()

    wav_files = sorted(
        (p for p in merged_root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"),
        key=lambda p: str(p).lower(),
    )
    summary.total_wav_files = len(wav_files)

    seen_sha: dict[str, str] = {}  # sha -> recording_id of kept original
    canonical_seen: dict[tuple[str, str], str] = {}  # (individual, canonical) -> recording_id
    used_ids: dict[str, int] = {}

    rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []

    for path in wav_files:
        parsed = _parse_record(path, merged_root, summary)
        if parsed is None:
            summary.skipped_non_audio.append(str(path))
            continue

        try:
            qc = _audio_qc(path)
        except Exception as exc:  # unreadable audio
            duplicate_rows.append(
                {
                    "raw_filename": parsed.get("raw_filename"),
                    "audio_path": parsed.get("audio_path"),
                    "duplicate_status": "unreadable",
                    "note": f"audio read failed: {exc}",
                }
            )
            summary.qc_fail += 1
            continue

        file_size = int(path.stat().st_size)
        sha = _sha256(path)

        duplicate_status = "unique"
        if sha in seen_sha:
            duplicate_status = "duplicate_exact"
            summary.exact_duplicates += 1
            duplicate_rows.append(
                {
                    "raw_filename": parsed["raw_filename"],
                    "audio_path": parsed["audio_path"],
                    "duplicate_status": duplicate_status,
                    "sha256": sha,
                    "kept_recording_id": seen_sha[sha],
                    "note": "identical content already kept",
                }
            )
            continue  # drop exact duplicates from the manifest

        canonical_key = (str(parsed["individual_id"]), str(parsed["canonical_filename"]))
        if canonical_key in canonical_seen:
            # Same canonical name, different content -> keep but flag.
            duplicate_status = "duplicate_candidate"
            summary.name_duplicates += 1
            duplicate_rows.append(
                {
                    "raw_filename": parsed["raw_filename"],
                    "audio_path": parsed["audio_path"],
                    "duplicate_status": duplicate_status,
                    "sha256": sha,
                    "kept_recording_id": canonical_seen[canonical_key],
                    "note": "same canonical name, different content (kept)",
                }
            )

        recording_id = _make_recording_id(str(parsed["canonical_filename"]), used_ids)
        seen_sha[sha] = recording_id
        canonical_seen.setdefault(canonical_key, recording_id)

        if parsed.get("_parse_failed"):
            qc_status, qc_notes = "warn", ["filename_unparsed"]
        else:
            qc_status, qc_notes = _qc_status(qc, qc_thresholds)
        if qc_status == "warn":
            summary.qc_warn += 1
        elif qc_status == "fail":
            summary.qc_fail += 1

        combined_notes = "; ".join(
            note for note in [str(parsed.get("notes", "")), "; ".join(qc_notes)] if note
        )

        rows.append(
            {
                "recording_id": recording_id,
                "individual_id": parsed["individual_id"],
                "session_id": parsed["session_id"],
                "song_type": parsed["song_type"],
                "context": parsed["context"],
                "audio_path": parsed["audio_path"],
                "temperature_log_path": "",
                "recording_start_utc": parsed["recording_start_utc"],
                "duration_s": qc["duration_s"],
                "sr": qc["sr"],
                "bit_depth": qc["bit_depth"],
                "notes": combined_notes,
                "group": parsed["group"],
                "daytime": parsed["daytime"],
                "female_context": parsed["female_context"],
                "sequence_id": parsed["sequence_id"],
                "date_iso": parsed["date_iso"],
                "time_hhmm": parsed["time_hhmm"],
                "canonical_filename": parsed["canonical_filename"],
                "raw_filename": parsed["raw_filename"],
                "sha256": sha,
                "file_size_bytes": file_size,
                "channels": qc["channels"],
                "rms_dbfs": round(float(qc["rms_dbfs"]), 3),
                "peak_dbfs": round(float(qc["peak_dbfs"]), 3),
                "clipping_ratio": round(float(qc["clipping_ratio"]), 6),
                "duplicate_status": duplicate_status,
                "qc_status": qc_status,
                "qc_notes": "; ".join(qc_notes),
            }
        )
        summary.kept += 1

    manifest_df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path / "recordings.csv", index=False)

    # No temperature logs available -> write an empty but well-formed manifest.
    pd.DataFrame(columns=TEMPERATURE_COLUMNS).to_csv(
        output_path / "temperature.csv", index=False
    )

    if interim_dir is not None:
        interim_path = Path(interim_dir)
        interim_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            duplicate_rows,
            columns=[
                "raw_filename",
                "audio_path",
                "duplicate_status",
                "sha256",
                "kept_recording_id",
                "note",
            ],
        ).to_csv(interim_path / "duplicate_report.csv", index=False)
        manifest_df[
            [
                "recording_id",
                "audio_path",
                "duration_s",
                "sr",
                "channels",
                "bit_depth",
                "rms_dbfs",
                "peak_dbfs",
                "clipping_ratio",
                "qc_status",
                "qc_notes",
            ]
        ].to_csv(interim_path / "qc_report.csv", index=False)

    return manifest_df, summary
