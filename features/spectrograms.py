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
    # Convert a (linear) power spectrum to decibels relative to its own peak:
    #   dB = 10 * log10(power / max(power)).
    # Output unit: dB (range <= 0, with 0 dB at the loudest bin). The 1e-10 floor
    # avoids log10(0) = -inf for silent bins.
    reference_power = max(float(np.max(power)), 1e-10)
    return (10.0 * np.log10(np.maximum(power, 1e-10))) - (10.0 * np.log10(reference_power))


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    # Hz -> mel using the HTK formula (mel = 2595 * log10(1 + Hz/700)).
    # Input unit: Hz. Output unit: mel (perceptual pitch scale, ~linear below 1 kHz).
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float32) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    # Inverse of _hz_to_mel. Input unit: mel. Output unit: Hz.
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float32) / 2595.0) - 1.0)


def _mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int,
    *,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    # freqs: the center frequency (Hz) of each FFT bin, from 0 Hz to the Nyquist
    # frequency sr/2. A real FFT of length n_fft yields n_fft//2 + 1 bins.
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1, dtype=np.float32)
    # Place n_mels+2 points equally spaced on the mel scale, then map back to Hz.
    # Equal spacing in mel -> triangular filters get wider (in Hz) at high
    # frequencies, matching human pitch perception. The +2 supplies the left/right
    # edges of the first and last triangle.
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2, dtype=np.float32)
    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros((n_mels, freqs.size), dtype=np.float32)
    for mel_index in range(n_mels):
        # Each mel band is a triangle (left, center, right) in Hz: a rising ramp
        # from left->center and a falling ramp from center->right. Multiplying the
        # filter bank (n_mels x n_bins) by the power spectrum sums FFT energy into
        # n_mels perceptual bands.
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

    # Short-Time Fourier Transform (STFT). All length parameters are in SAMPLES,
    # not seconds or Hz:
    #   nperseg / nfft = n_fft  -> FFT window length in samples (e.g. 1024 @ 44.1 kHz
    #                              = ~23.2 ms, giving ~43 Hz frequency resolution).
    #   noverlap = n_fft - hop_length -> samples shared between consecutive frames;
    #                              hop_length is the step (in samples) between frames.
    # zxx is the complex STFT of shape (n_freq_bins, n_frames).
    _, _, zxx = stft(
        audio_array,
        fs=sr,
        nperseg=int(spectrogram_config["n_fft"]),
        noverlap=int(spectrogram_config["n_fft"]) - int(spectrogram_config["hop_length"]),
        nfft=int(spectrogram_config["n_fft"]),
        boundary="zeros",
        padded=True,
    )
    # power = 1.0 -> magnitude spectrum |X|; power = 2.0 -> power spectrum |X|^2
    # (the MVP default; energy is what the mel filter bank integrates).
    power_spectrum = np.abs(zxx) ** float(spectrogram_config["power"])
    mel_filter = _mel_filter_bank(
        sr,
        int(spectrogram_config["n_fft"]),
        int(spectrogram_config["n_mels"]),
        fmin=0.0,
        fmax=float(sr) / 2.0,
    )
    # (n_mels x n_bins) @ (n_bins x n_frames) -> (n_mels x n_frames) mel power.
    mel_power = np.maximum(mel_filter @ power_spectrum, 1e-10)
    mel_db = _power_to_db(mel_power)  # log-mel spectrogram, unit: dB
    # Per-spectrogram min-max normalization to [0, 1] so every CNN input shares
    # the same numeric range regardless of absolute recording loudness. (This is
    # numeric normalization, NOT loudness normalization of the raw audio.)
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
