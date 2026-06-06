from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct
from scipy.signal import stft

from cricket_id.io.manifest import _write_parquet_compatible
from cricket_id.utils.paths import resolve_data_dir


def _segment_audio_path(segment_id: str, data_dir: Path) -> Path:
    return data_dir / "processed" / "segments_wav" / f"{segment_id}.wav"


def _frame_audio(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)
    if audio_array.size < frame_length:
        audio_array = np.pad(audio_array, (0, frame_length - audio_array.size))

    frame_count = 1 + max(0, (audio_array.size - frame_length) // hop_length)
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    return np.stack([audio_array[start : start + frame_length] for start in starts], axis=0)


def _summary_stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _compute_delta(values: np.ndarray) -> np.ndarray:
    # Delta features = first-order temporal derivative of the MFCC trajectory,
    # approximated frame-to-frame with a central difference (np.gradient over the
    # time axis). Captures how the spectral envelope CHANGES over the chirp.
    if values.shape[1] < 2:
        return np.zeros_like(values)
    return np.gradient(values, axis=1).astype(np.float32, copy=False)


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


def _power_to_db(power: np.ndarray) -> np.ndarray:
    reference_power = max(float(np.max(power)), 1e-10)
    return (10.0 * np.log10(np.maximum(power, 1e-10))) - (10.0 * np.log10(reference_power))


def _spectral_rolloff(power_spectrum: np.ndarray, freqs: np.ndarray, roll_percent: float) -> np.ndarray:
    # Spectral roll-off: per frame, the frequency (Hz) below which `roll_percent`
    # (here 85 %) of the total spectral energy is contained. Found by walking the
    # cumulative energy until it crosses the threshold. Output unit: Hz per frame.
    total_energy = np.sum(power_spectrum, axis=0)
    thresholds = total_energy * roll_percent
    cumulative = np.cumsum(power_spectrum, axis=0)
    rolloff = np.zeros(power_spectrum.shape[1], dtype=np.float32)
    for frame_index in range(power_spectrum.shape[1]):
        if thresholds[frame_index] <= 0:
            continue
        target_index = int(np.searchsorted(cumulative[:, frame_index], thresholds[frame_index]))
        target_index = min(target_index, freqs.size - 1)
        rolloff[frame_index] = float(freqs[target_index])
    return rolloff


def _compute_time_frequency_representation(
    audio: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    # STFT length parameters are in SAMPLES (win_length / n_fft / hop_length);
    # `fs=sr` (Hz) is only used so scipy returns `freqs` already in Hz.
    #   nperseg = win_length -> analysis window length (samples)
    #   nfft    = n_fft      -> zero-padded FFT length (samples); >= win_length
    #   noverlap = win_length - hop_length -> overlap (samples) between frames
    freqs, _, zxx = stft(
        np.asarray(audio, dtype=np.float32),
        fs=sr,
        nperseg=win_length,
        noverlap=max(0, win_length - hop_length),
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.abs(zxx).astype(np.float32, copy=False)  # |X|, linear amplitude
    power = np.square(magnitude, dtype=np.float32)           # |X|^2, power spectrum
    return freqs.astype(np.float32, copy=False), power  # freqs in Hz, power per (bin, frame)


def extract_features_for_segment(audio: np.ndarray, sr: int, config: dict) -> dict:
    features_config = config["features"]
    audio_array = np.asarray(audio, dtype=np.float32)

    n_fft = int(features_config["n_fft"])
    hop_length = int(features_config["hop_length"])
    win_length = int(features_config["win_length"])
    n_mfcc = int(features_config["n_mfcc"])
    n_mels = int(features_config["n_mels"])
    fmin_hz = float(features_config["fmin_hz"])
    fmax_hz = float(features_config["fmax_hz"])

    frames = _frame_audio(audio_array, win_length, hop_length)
    # RMS (root-mean-square) amplitude per frame: a per-frame loudness/energy
    # proxy in linear amplitude units (unitless, since audio is in [-1, 1]).
    rms = np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32, copy=False)
    # Zero-crossing rate: fraction of adjacent samples whose sign flips within a
    # frame (np.signbit detects sign, np.diff counts flips). Dimensionless in
    # [0, 1]; higher ZCR ~ more high-frequency / noisy content.
    zcr = (
        np.sum(np.diff(np.signbit(frames), axis=1), axis=1).astype(np.float32)
        / max(1, win_length - 1)
    )

    freqs, power_spectrum = _compute_time_frequency_representation(
        audio_array,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    magnitude_spectrum = np.sqrt(np.maximum(power_spectrum, 0.0)).astype(np.float32, copy=False)
    magnitude_sum = np.sum(magnitude_spectrum, axis=0)
    safe_magnitude_sum = np.where(magnitude_sum > 0, magnitude_sum, 1.0)

    # Spectral centroid (Hz): magnitude-weighted mean frequency per frame, i.e.
    # the spectral "center of mass" / perceived brightness.
    spectral_centroid = (
        np.sum(freqs[:, None] * magnitude_spectrum, axis=0) / safe_magnitude_sum
    ).astype(np.float32, copy=False)
    # Spectral bandwidth (Hz): magnitude-weighted standard deviation of frequency
    # about the centroid -> how spread out the spectrum is per frame.
    spectral_bandwidth = np.sqrt(
        np.sum(
            ((freqs[:, None] - spectral_centroid[None, :]) ** 2) * magnitude_spectrum,
            axis=0,
        )
        / safe_magnitude_sum
    ).astype(np.float32, copy=False)
    spectral_rolloff = _spectral_rolloff(power_spectrum, freqs, roll_percent=0.85)
    # Spectral flatness (dimensionless, 0..1): geometric mean / arithmetic mean of
    # the magnitude spectrum. ~1 = noise-like (flat), ~0 = tonal (peaky).
    spectral_flatness = (
        np.exp(np.mean(np.log(np.maximum(magnitude_spectrum, 1e-10)), axis=0))
        / np.maximum(np.mean(magnitude_spectrum, axis=0), 1e-10)
    ).astype(np.float32, copy=False)

    mel_filter = _mel_filter_bank(
        sr,
        n_fft,
        n_mels,
        fmin=fmin_hz,
        fmax=fmax_hz,
    )
    mel_power = np.maximum(mel_filter @ power_spectrum, 1e-10)
    log_mel = _power_to_db(mel_power).astype(np.float32, copy=False)  # log-mel, unit: dB
    # MFCCs = type-2 DCT of the log-mel spectrum, keeping the first n_mfcc (13)
    # coefficients. The DCT decorrelates the mel bands and compresses the spectral
    # envelope; norm="ortho" makes the transform energy-preserving. Coefficient 0
    # ~ overall log-energy, higher coefficients ~ finer spectral detail.
    mfcc = dct(log_mel, axis=0, type=2, norm="ortho")[:n_mfcc].astype(np.float32, copy=False)
    delta_mfcc = _compute_delta(mfcc)

    mean_spectrum = magnitude_spectrum.mean(axis=1) if magnitude_spectrum.size else np.zeros(n_fft // 2 + 1)
    if np.allclose(mean_spectrum, 0.0):
        peak_freq_hz = 0.0
        bandwidth_hz_proxy = 0.0
    else:
        # peak_freq_hz: dominant frequency (Hz) = bin with the highest time-averaged
        # magnitude. For crickets this tracks the carrier frequency of the chirp.
        peak_index = int(np.argmax(mean_spectrum))
        peak_freq_hz = float(freqs[peak_index])
        # bandwidth_hz_proxy: width (Hz) of the band whose averaged magnitude is at
        # least half the peak (the -6 dB-in-amplitude / "half-power" span).
        half_power = float(np.max(mean_spectrum) * 0.5)
        active_bins = np.flatnonzero(mean_spectrum >= half_power)
        bandwidth_hz_proxy = (
            float(freqs[active_bins[-1]] - freqs[active_bins[0]])
            if active_bins.size
            else 0.0
        )

    feature_dict: dict[str, float] = {
        "duration_ms": float(audio_array.shape[0] / sr * 1000.0),
        "peak_freq_hz": peak_freq_hz,
        "bandwidth_hz_proxy": bandwidth_hz_proxy,
    }

    summary_inputs = {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
    }
    for prefix, values in summary_inputs.items():
        mean_value, std_value = _summary_stats(values)
        feature_dict[f"{prefix}_mean"] = mean_value
        feature_dict[f"{prefix}_std"] = std_value

    for index in range(n_mfcc):
        mean_value, std_value = _summary_stats(mfcc[index])
        feature_dict[f"mfcc_{index + 1}_mean"] = mean_value
        feature_dict[f"mfcc_{index + 1}_std"] = std_value

    for index in range(n_mfcc):
        mean_value, std_value = _summary_stats(delta_mfcc[index])
        feature_dict[f"delta_mfcc_{index + 1}_mean"] = mean_value
        feature_dict[f"delta_mfcc_{index + 1}_std"] = std_value

    return feature_dict


def run_feature_extraction(
    segments_df: pd.DataFrame,
    config: dict,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    data_dir = resolve_data_dir(config, project_root=project_root)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, object]] = []
    for row in segments_df.itertuples(index=False):
        segment_path = _segment_audio_path(str(row.segment_id), data_dir)
        audio, sr = sf.read(segment_path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        features = extract_features_for_segment(audio, sr, config)
        feature_rows.append(
            {
                "segment_id": row.segment_id,
                "individual_id": row.individual_id,
                "session_id": row.session_id,
                "recording_id": row.recording_id,
                "temperature_c": row.temperature_c,
                **features,
            }
        )

    features_df = pd.DataFrame(feature_rows)
    metadata_columns = {
        "segment_id",
        "individual_id",
        "session_id",
        "recording_id",
        "temperature_c",
    }
    if features_df.empty:
        base_columns = [
            "segment_id",
            "individual_id",
            "session_id",
            "recording_id",
            "temperature_c",
        ]
        sample_feature_columns = list(
            extract_features_for_segment(
                np.zeros(
                    int(
                        round(
                            float(config["segmentation"]["fixed_duration_s"])
                            * int(config.get("audio", {}).get("target_sr", 44100))
                        )
                    ),
                    dtype=np.float32,
                ),
                int(config.get("audio", {}).get("target_sr", 44100)),
                config,
            ).keys()
        )
        features_df = pd.DataFrame(columns=base_columns + sample_feature_columns)

    feature_columns = [column for column in features_df.columns if column not in metadata_columns]
    if feature_columns and features_df[feature_columns].isna().any().any():
        nan_columns = features_df[feature_columns].columns[
            features_df[feature_columns].isna().any()
        ].tolist()
        raise AssertionError(f"NaN values found in feature columns: {', '.join(nan_columns)}")

    output_path = processed_dir / "features_tabular.parquet"
    _write_parquet_compatible(features_df, output_path)
    return features_df
