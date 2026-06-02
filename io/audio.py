from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from cricket_id.io.manifest import _write_parquet_compatible


def load_audio(path: str, target_sr: int = 44100) -> tuple[np.ndarray, int]:
    audio, original_sr = sf.read(path, always_2d=False)
    audio_array = np.asarray(audio, dtype=np.float32)

    if audio_array.ndim == 1:
        mono_audio = audio_array
    else:
        mono_audio = librosa.to_mono(audio_array.T)

    if original_sr != target_sr:
        mono_audio = librosa.resample(
            mono_audio,
            orig_sr=original_sr,
            target_sr=target_sr,
        )

    return np.asarray(mono_audio, dtype=np.float32), target_sr


def compute_audio_qc(
    audio: np.ndarray,
    sr: int,
    original_sr: int,
    original_duration: float,
) -> dict[str, int | float | bool]:
    peak_amplitude = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "original_sr": int(original_sr),
        "target_sr": int(sr),
        "original_duration_s": float(original_duration),
        "n_samples": int(audio.shape[0]),
        "peak_amplitude": peak_amplitude,
        "clipping_flag": bool(peak_amplitude > 0.99),
    }


def build_audio_index(recordings_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_dir = Path(config["paths"]["data_dir"])
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    target_sr = int(config.get("audio", {}).get("target_sr", 44100))
    qc_rows: list[dict[str, object]] = []

    for row in recordings_df.itertuples(index=False):
        audio_path = Path(str(row.audio_path))
        resolved_audio_path = audio_path if audio_path.is_absolute() else raw_dir / audio_path
        audio, sr = load_audio(str(resolved_audio_path), target_sr=target_sr)
        qc = compute_audio_qc(
            audio=audio,
            sr=sr,
            original_sr=int(row.sr),
            original_duration=float(row.duration_s),
        )
        qc_rows.append({"recording_id": row.recording_id, **qc})

    audio_index_df = pd.DataFrame(qc_rows)
    _write_parquet_compatible(audio_index_df, interim_dir / "audio_index.parquet")
    return audio_index_df
