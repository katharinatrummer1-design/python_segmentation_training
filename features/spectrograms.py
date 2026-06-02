from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly, stft

from cricket_id.io.manifest import _write_parquet_compatible
from cricket_id.utils.paths import resolve_data_dir


def _segment_audio_path(segment_id: str, data_dir: Path) -> Path:
    return data_dir / "processed" / "segments_wav" / f"{segment_id}.wav"


def _power_to_db(power: np.ndarray) -> np.ndarray:
    reference_power = max(float(np.max(power)), 1e-10)
    return (10.0 * np.log10(np.maximum(power, 1e-10))) - (10.0 * np.log10(reference_power))


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float32) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float32) / 2595.0) - 1.0)


def _mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int,
    *,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1, dtype=np.float32)
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2, dtype=np.float32)
    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for mel_index in range(n_mels):
        left = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        right = hz_points[mel_index + 2]
        if center <= left or right <= center:
            continue

        rising = (freqs >= left) & (freqs <= center)
        falling = (freqs >= center) & (freqs <= right)
        filters[mel_index, rising] = (freqs[rising] - left) / max(center - left, 1e-10)
        filters[mel_index, falling] = (right - freqs[falling]) / max(right - center, 1e-10)

    return filters


def compute_log_mel_spectrogram(audio: np.ndarray, sr: int, config: dict) -> np.ndarray:
    spectrogram_config = config["spectrogram"]
    audio_array = np.asarray(audio, dtype=np.float32)
    target_sr = int(spectrogram_config.get("sr", sr))
    if sr != target_sr:
        audio_array = resample_poly(audio_array, target_sr, sr).astype(np.float32, copy=False)
        sr = target_sr

    _, _, zxx = stft(
        audio_array,
        fs=sr,
        nperseg=int(spectrogram_config["n_fft"]),
        noverlap=int(spectrogram_config["n_fft"]) - int(spectrogram_config["hop_length"]),
        nfft=int(spectrogram_config["n_fft"]),
        boundary="zeros",
        padded=True,
    )
    power_spectrum = np.abs(zxx) ** float(spectrogram_config["power"])
    mel_filter = _mel_filter_bank(
        sr,
        int(spectrogram_config["n_fft"]),
        int(spectrogram_config["n_mels"]),
        fmin=0.0,
        fmax=float(sr) / 2.0,
    )
    mel_power = np.maximum(mel_filter @ power_spectrum, 1e-10)
    mel_db = _power_to_db(mel_power)
    mel_min = float(np.min(mel_db))
    mel_max = float(np.max(mel_db))
    if np.isclose(mel_max, mel_min):
        return np.zeros_like(mel_db, dtype=np.float32)
    normalized = (mel_db - mel_min) / (mel_max - mel_min)
    return normalized.astype(np.float32, copy=False)


def run_spectrogram_extraction(
    segments_df: pd.DataFrame,
    config: dict,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    data_dir = resolve_data_dir(config, project_root=project_root)
    output_dir = data_dir / "processed" / "spectrograms"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    shapes: set[tuple[int, int]] = set()
    for row in segments_df.itertuples(index=False):
        segment_path = _segment_audio_path(str(row.segment_id), data_dir)
        audio, sr = sf.read(segment_path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        spectrogram = compute_log_mel_spectrogram(audio, sr, config)
        npy_path = output_dir / f"{row.segment_id}.npy"
        np.save(npy_path, spectrogram)
        shapes.add(tuple(int(value) for value in spectrogram.shape))

        rows.append(
            {
                "segment_id": row.segment_id,
                "individual_id": row.individual_id,
                "session_id": row.session_id,
                "npy_path": str(npy_path),
                "shape_0": int(spectrogram.shape[0]),
                "shape_1": int(spectrogram.shape[1]),
            }
        )

    if len(shapes) > 1:
        raise AssertionError(f"Spectrogram shapes are inconsistent: {sorted(shapes)}")

    index_columns = [
        "segment_id",
        "individual_id",
        "session_id",
        "npy_path",
        "shape_0",
        "shape_1",
    ]
    index_df = pd.DataFrame(rows, columns=index_columns)
    output_path = data_dir / "processed" / "spectrogram_index.parquet"
    _write_parquet_compatible(index_df, output_path)
    return index_df
