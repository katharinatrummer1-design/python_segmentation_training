from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt, sosfiltfilt

from cricket_id.io.audio import load_audio
from cricket_id.io.manifest import _write_parquet_compatible
from cricket_id.utils.paths import (
    first_existing_path_hint,
    resolve_audio_path,
    resolve_data_dir,
)


def _compute_frame_rms(
    audio: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(1, dtype=np.float32)

    working_audio = np.asarray(audio, dtype=np.float32)
    if working_audio.size < frame_length:
        working_audio = np.pad(
            working_audio,
            (0, frame_length - working_audio.size),
            mode="constant",
        )

    frame_count = 1 + max(0, (working_audio.size - frame_length) // hop_length)
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    frames = np.stack(
        [working_audio[start : start + frame_length] for start in starts],
        axis=0,
    )
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32, copy=False)


def _resolve_segment_directories(
    config: dict,
    *,
    recordings_qc_df: pd.DataFrame | None = None,
    project_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    path_hint = None
    if recordings_qc_df is not None:
        path_hint = first_existing_path_hint(recordings_qc_df, "audio_path")

    data_dir = resolve_data_dir(config, project_root=project_root, path_hint=path_hint)
    interim_dir = data_dir / "interim"
    segments_dir = data_dir / "processed" / "segments_wav"
    interim_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, interim_dir, segments_dir


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    fmin_hz: float,
    fmax_hz: float,
) -> np.ndarray:
    if audio.size == 0:
        return np.asarray(audio, dtype=np.float32)

    nyquist_hz = sr / 2.0
    low_hz = max(1.0, float(fmin_hz))
    high_hz = min(float(fmax_hz), nyquist_hz * 0.99)
    if low_hz >= high_hz:
        return np.asarray(audio, dtype=np.float32).copy()

    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    filtered = sosfilt(sos, np.asarray(audio, dtype=np.float32))
    return filtered.astype(np.float32, copy=False)


def highpass_filter(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    *,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply a high-pass filter (default zero-phase / filtfilt).

    Per FEATURES_BEREINIGUNG_GRILLEN.md the standard filter is a high-pass that
    removes low-frequency rumble while leaving the upper spectrum intact (no
    band-pass, no upper cut), so fine individual spectral cues are preserved.
    """
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.size == 0:
        return audio_array

    nyquist_hz = sr / 2.0
    cutoff = max(1.0, float(cutoff_hz))
    if cutoff >= nyquist_hz * 0.99:
        return audio_array.copy()

    sos = butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    if zero_phase and audio_array.size > 3 * (order + 1):
        filtered = sosfiltfilt(sos, audio_array)
    else:
        filtered = sosfilt(sos, audio_array)
    return np.asarray(filtered, dtype=np.float32)


def apply_audio_filter(audio: np.ndarray, sr: int, config: dict) -> np.ndarray:
    """Select and apply the configured analysis filter.

    Controlled by the ``audio_filter`` config block; falls back to the legacy
    ``bandpass`` block when no ``audio_filter`` is configured.
    """
    filter_config = config.get("audio_filter")
    if filter_config:
        filter_type = str(filter_config.get("type", "highpass")).lower()
        if filter_type == "highpass":
            return highpass_filter(
                audio,
                sr,
                cutoff_hz=float(filter_config.get("cutoff_hz", 500.0)),
                order=int(filter_config.get("order", 4)),
                zero_phase=bool(filter_config.get("zero_phase", True)),
            )
        if filter_type == "bandpass":
            return bandpass_filter(
                audio,
                sr,
                fmin_hz=float(filter_config.get("fmin_hz", 2000.0)),
                fmax_hz=float(filter_config.get("fmax_hz", 8000.0)),
            )
        if filter_type in {"none", "off", "disabled"}:
            return np.asarray(audio, dtype=np.float32)

    bandpass_config = config.get("bandpass")
    if bandpass_config:
        return bandpass_filter(
            audio,
            sr,
            fmin_hz=float(bandpass_config["fmin_hz"]),
            fmax_hz=float(bandpass_config["fmax_hz"]),
        )
    return np.asarray(audio, dtype=np.float32)


def detect_segments(audio: np.ndarray, sr: int, config: dict) -> list[dict]:
    segmentation_config = config["segmentation"]
    frame_length = int(segmentation_config["rms_frame_length"])
    hop_length = int(segmentation_config["rms_hop_length"])
    merge_gap_s = float(segmentation_config["merge_gap_s"])
    min_duration_s = float(segmentation_config["min_duration_s"])
    max_duration_s = float(segmentation_config["max_duration_s"])
    threshold_db_below_peak = float(segmentation_config["threshold_db_below_peak"])

    if audio.size == 0:
        return []

    rms = _compute_frame_rms(
        np.asarray(audio, dtype=np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    noise_floor = float(np.percentile(rms, 10)) if rms.size else 0.0

    if peak_rms <= 0.0:
        return [
            {
                "start_s": 0.0,
                "end_s": float(audio.shape[0] / sr),
                "duration_s_raw": float(audio.shape[0] / sr),
            }
        ]

    relative_threshold = peak_rms * (10.0 ** (threshold_db_below_peak / 20.0))
    threshold = max(noise_floor, relative_threshold)
    active = rms >= threshold

    regions: list[tuple[int, int]] = []
    start_frame: int | None = None
    for frame_index, is_active in enumerate(active):
        if is_active and start_frame is None:
            start_frame = frame_index
        elif not is_active and start_frame is not None:
            regions.append((start_frame, frame_index))
            start_frame = None
    if start_frame is not None:
        regions.append((start_frame, len(active)))

    sample_regions: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        start_sample = region_start * hop_length
        end_sample = min(audio.shape[0], (region_end - 1) * hop_length + frame_length)
        if end_sample > start_sample:
            sample_regions.append((start_sample, end_sample))

    merged_regions: list[tuple[int, int]] = []
    merge_gap_samples = int(round(merge_gap_s * sr))
    for start_sample, end_sample in sample_regions:
        if not merged_regions:
            merged_regions.append((start_sample, end_sample))
            continue

        previous_start, previous_end = merged_regions[-1]
        if start_sample - previous_end <= merge_gap_samples:
            merged_regions[-1] = (previous_start, max(previous_end, end_sample))
        else:
            merged_regions.append((start_sample, end_sample))

    segments: list[dict] = []
    for start_sample, end_sample in merged_regions:
        duration_s = (end_sample - start_sample) / sr
        if duration_s < min_duration_s or duration_s > max_duration_s:
            continue
        segments.append(
            {
                "start_s": start_sample / sr,
                "end_s": end_sample / sr,
                "duration_s_raw": duration_s,
            }
        )

    if not segments:
        full_duration_s = float(audio.shape[0] / sr)
        return [
            {
                "start_s": 0.0,
                "end_s": full_duration_s,
                "duration_s_raw": full_duration_s,
            }
        ]

    return segments


def standardize_segment(audio: np.ndarray, sr: int, fixed_duration_s: float) -> np.ndarray:
    target_samples = int(round(float(fixed_duration_s) * sr))
    segment_audio = np.asarray(audio, dtype=np.float32)

    if target_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    if segment_audio.shape[0] == target_samples:
        return segment_audio.copy()

    if segment_audio.shape[0] < target_samples:
        deficit = target_samples - segment_audio.shape[0]
        pad_left = deficit // 2
        pad_right = deficit - pad_left
        return np.pad(segment_audio, (pad_left, pad_right), mode="constant")

    excess = segment_audio.shape[0] - target_samples
    crop_left = excess // 2
    crop_right = crop_left + target_samples
    return segment_audio[crop_left:crop_right].astype(np.float32, copy=False)


def compute_snr_proxy(segment_audio: np.ndarray, full_audio: np.ndarray) -> float:
    segment_array = np.asarray(segment_audio, dtype=np.float32)
    if segment_array.size == 0:
        return 0.0

    full_array = np.asarray(full_audio, dtype=np.float32)
    full_size = int(full_array.shape[0])
    segment_rms = float(np.sqrt(np.mean(np.square(segment_array))))
    full_rms = _compute_frame_rms(
        full_array,
        frame_length=min(1024, max(32, full_size)),
        hop_length=512 if full_size >= 512 else max(1, full_size // 2 or 1),
    )
    noise_floor = float(np.percentile(full_rms, 10)) if full_rms.size else 0.0
    eps = 1e-10
    return float(20.0 * np.log10(max(segment_rms, eps) / max(noise_floor, eps)))


def _build_qc_flag(segment_audio: np.ndarray, full_audio: np.ndarray, snr_proxy: float) -> str:
    flags: list[str] = []
    segment_peak = float(np.max(np.abs(segment_audio))) if np.asarray(segment_audio).size else 0.0
    full_peak = float(np.max(np.abs(full_audio))) if np.asarray(full_audio).size else 0.0
    if segment_peak >= 0.99 or full_peak >= 0.99:
        flags.append("clipping")
    if snr_proxy < 3.0:
        flags.append("low_snr")
    return "ok" if not flags else ";".join(flags)


def run_segmentation(
    recordings_qc_df: pd.DataFrame,
    config: dict,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    _, interim_dir, segments_dir = _resolve_segment_directories(
        config,
        recordings_qc_df=recordings_qc_df,
        project_root=project_root,
    )

    target_sr = int(config.get("audio", {}).get("target_sr", 44100))
    fixed_duration_s = float(config["segmentation"]["fixed_duration_s"])

    segment_columns = [
        "segment_id",
        "recording_id",
        "individual_id",
        "session_id",
        "start_s",
        "end_s",
        "duration_s_raw",
        "duration_s_fixed",
        "temperature_c",
        "snr_proxy",
        "qc_flag",
    ]
    segment_rows: list[dict[str, object]] = []
    for row in recordings_qc_df.itertuples(index=False):
        audio_path = resolve_audio_path(
            getattr(row, "audio_path"),
            config,
            project_root=project_root,
            path_hint=getattr(row, "audio_path", None),
        )
        audio, sr = load_audio(str(audio_path), target_sr=target_sr)
        filtered_audio = apply_audio_filter(audio, sr, config)
        segments = detect_segments(filtered_audio, sr, config)

        max_segments = config.get("segmentation", {}).get("max_segments_per_recording")
        if max_segments is not None and len(segments) > int(max_segments):
            # Deterministically keep an evenly spaced subset across the
            # recording so we sample the whole file rather than just its start.
            keep = int(max_segments)
            indices = np.linspace(0, len(segments) - 1, keep).round().astype(int)
            indices = sorted(set(indices.tolist()))
            segments = [segments[i] for i in indices]

        for index, segment in enumerate(segments):
            start_sample = max(0, int(round(float(segment["start_s"]) * sr)))
            end_sample = min(filtered_audio.shape[0], int(round(float(segment["end_s"]) * sr)))
            raw_segment_audio = filtered_audio[start_sample:end_sample]
            fixed_audio = standardize_segment(raw_segment_audio, sr, fixed_duration_s)
            segment_id = f"{row.recording_id}_seg{index:04d}"
            segment_path = segments_dir / f"{segment_id}.wav"
            sf.write(segment_path, fixed_audio, sr)

            snr_proxy = compute_snr_proxy(raw_segment_audio, filtered_audio)
            segment_rows.append(
                {
                    "segment_id": segment_id,
                    "recording_id": row.recording_id,
                    "individual_id": row.individual_id,
                    "session_id": row.session_id,
                    "start_s": float(segment["start_s"]),
                    "end_s": float(segment["end_s"]),
                    "duration_s_raw": float(segment["duration_s_raw"]),
                    "duration_s_fixed": float(fixed_duration_s),
                    "temperature_c": float(
                        getattr(row, "temperature_c", getattr(row, "temp_mean_c", np.nan))
                    ),
                    "snr_proxy": float(snr_proxy),
                    "qc_flag": _build_qc_flag(fixed_audio, filtered_audio, snr_proxy),
                }
            )

    segments_df = pd.DataFrame(segment_rows, columns=segment_columns)
    output_path = interim_dir / "segments.parquet"
    _write_parquet_compatible(segments_df, output_path)
    return segments_df
