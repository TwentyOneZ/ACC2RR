from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

EPS = np.finfo(float).eps

@dataclass
class Config:
    fmin_hz: float = 0.08
    fmax_hz: float = 0.80
    filter_order: int = 4
    hampel_window_s: float = 0.40
    hampel_n_sigma: float = 4.0
    max_gap_s: float = 0.50
    gap_factor: float = 1.5
    window_s: float = 30.0
    hop_s: float = 1.0
    welch_segment_s: float = 60.0
    local_peak_halfwidth_hz: float = 0.03
    harmonic_half_power_ratio: float = 0.20
    consensus_agreement_bpm: float = 3.0
    min_confidence: float = 0.45
    kalman_process_var_bpm2_per_s: float = 0.25
    kalman_base_measurement_var_bpm2: float = 4.0


@dataclass
class TimestampQC:
    raw_rows: int
    usable_rows: int
    duplicate_timestamps: int
    nonpositive_dt: int
    median_dt_s: float
    effective_fs_hz: float
    nominal_fs_hz: float
    dt_jitter_ms: float
    short_gap_count: int
    long_gap_count: int
    estimated_missing_samples: int
    selected_duration_s: float
    excluded_duration_s: float
    timestamp_quality: float


@dataclass
class Estimate:
    rr_psd_raw_bpm: float
    rr_psd_bpm: float
    rr_acf_bpm: float
    rr_final_bpm: float
    spectral_quality: float
    acf_quality: float
    agreement_quality: float
    consensus_score: float
    harmonic_correction_applied: bool
    pca_pc1_variance_ratio: float
    pca_eigenvalues: list[float]
    confidence: float


def _finite_float(x: Any) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def load_metadata(recording_dir: Path) -> dict[str, Any]:
    path = recording_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return {}


def read_acc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp_epoch_s", "x_mg", "y_mg", "z_mg"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df[["timestamp_epoch_s", "x_mg", "y_mg", "z_mg"]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    if len(out) < 20:
        raise ValueError(f"{path} contains too few usable accelerometer rows ({len(out)}).")
    return out


def timestamp_qc_and_segment(
    df: pd.DataFrame,
    nominal_fs_hz: float | None,
    cfg: Config,
) -> tuple[pd.DataFrame, TimestampQC]:
    raw_rows = len(df)
    df = df.sort_values("timestamp_epoch_s", kind="stable").reset_index(drop=True)

    duplicate_timestamps = int(df["timestamp_epoch_s"].duplicated().sum())
    if duplicate_timestamps:
        df = (
            df.groupby("timestamp_epoch_s", as_index=False)[["x_mg", "y_mg", "z_mg"]]
            .mean()
            .sort_values("timestamp_epoch_s")
            .reset_index(drop=True)
        )

    t = df["timestamp_epoch_s"].to_numpy(float)
    dt = np.diff(t)
    nonpositive = int(np.sum(dt <= 0))
    positive_dt = dt[dt > 0]
    if positive_dt.size == 0:
        raise ValueError("No positive timestamp intervals were found.")

    median_dt = float(np.median(positive_dt))
    effective_fs = 1.0 / median_dt
    if nominal_fs_hz is None or not np.isfinite(nominal_fs_hz) or nominal_fs_hz <= 0:
        nominal_fs = effective_fs
    else:
        nominal_fs = float(nominal_fs_hz)
    target_dt = 1.0 / nominal_fs

    jitter_ms = float(1000.0 * np.median(np.abs(positive_dt - median_dt)))
    gap_threshold = cfg.gap_factor * target_dt
    short_gap_mask = (dt > gap_threshold) & (dt <= cfg.max_gap_s)
    long_gap_mask = dt > cfg.max_gap_s
    short_gap_count = int(short_gap_mask.sum())
    long_gap_count = int(long_gap_mask.sum())
    estimated_missing = int(
        np.sum(np.maximum(np.rint(positive_dt / target_dt).astype(int) - 1, 0))
    )

    full_duration = float(t[-1] - t[0])
    excluded_duration = 0.0

    # Never interpolate across a long discontinuity. Keep the longest contiguous
    # segment, which is safer for low-frequency respiratory analysis.
    if long_gap_count:
        split_after = np.where(long_gap_mask)[0]
        starts = np.r_[0, split_after + 1]
        stops = np.r_[split_after + 1, len(df)]
        lengths = stops - starts
        best = int(np.argmax(lengths))
        df = df.iloc[starts[best] : stops[best]].reset_index(drop=True)
        t_selected = df["timestamp_epoch_s"].to_numpy(float)
        selected_duration = float(t_selected[-1] - t_selected[0])
        excluded_duration = max(0.0, full_duration - selected_duration)
    else:
        selected_duration = full_duration

    # Timestamp quality is heuristic and is reported explicitly rather than used
    # as a hidden hard rejection criterion.
    missing_fraction = estimated_missing / max(raw_rows + estimated_missing, 1)
    jitter_ratio = jitter_ms / max(target_dt * 1000.0, EPS)
    long_gap_penalty = min(1.0, long_gap_count * 0.25)
    timestamp_quality = float(
        np.clip(1.0 - 2.0 * missing_fraction - 0.5 * jitter_ratio - long_gap_penalty, 0.0, 1.0)
    )

    qc = TimestampQC(
        raw_rows=raw_rows,
        usable_rows=len(df),
        duplicate_timestamps=duplicate_timestamps,
        nonpositive_dt=nonpositive,
        median_dt_s=median_dt,
        effective_fs_hz=effective_fs,
        nominal_fs_hz=nominal_fs,
        dt_jitter_ms=jitter_ms,
        short_gap_count=short_gap_count,
        long_gap_count=long_gap_count,
        estimated_missing_samples=estimated_missing,
        selected_duration_s=selected_duration,
        excluded_duration_s=excluded_duration,
        timestamp_quality=timestamp_quality,
    )
    return df, qc


def resample_uniform(df: pd.DataFrame, fs_hz: float) -> tuple[np.ndarray, np.ndarray, float]:
    t_abs = df["timestamp_epoch_s"].to_numpy(float)
    xyz = df[["x_mg", "y_mg", "z_mg"]].to_numpy(float)
    dt = 1.0 / fs_hz
    t0, t1 = float(t_abs[0]), float(t_abs[-1])
    t_uniform_abs = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
    xyz_uniform = np.column_stack(
        [np.interp(t_uniform_abs, t_abs, xyz[:, axis]) for axis in range(3)]
    )
    return t_uniform_abs - t_uniform_abs[0], xyz_uniform, t0


def hampel_vector_despike(
    xyz: np.ndarray,
    fs_hz: float,
    window_s: float,
    n_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    window = max(3, int(round(window_s * fs_hz)))
    if window % 2 == 0:
        window += 1

    mask = np.zeros(len(xyz), dtype=bool)
    for axis in range(3):
        s = pd.Series(xyz[:, axis])
        med = s.rolling(window, center=True, min_periods=max(3, window // 3)).median()
        deviation = (s - med).abs()
        mad = deviation.rolling(
            window, center=True, min_periods=max(3, window // 3)
        ).median()
        scale = 1.4826 * mad.to_numpy(float)
        dev = deviation.to_numpy(float)

        # Avoid classifying ordinary quantization as spikes when local MAD is zero.
        positive_scale = scale[np.isfinite(scale) & (scale > 0)]
        floor = float(np.median(positive_scale) * 0.10) if positive_scale.size else 1.0
        floor = max(floor, 1.0)  # Polar H10 values are in mg integer units.
        mask |= dev > n_sigma * np.maximum(scale, floor)

    clean = xyz.copy()
    indices = np.arange(len(xyz), dtype=float)
    good = ~mask
    if mask.any() and good.sum() >= 2:
        # Replace the whole XYZ vector whenever any coordinate is an outlier.
        for axis in range(3):
            clean[mask, axis] = np.interp(indices[mask], indices[good], xyz[good, axis])
    return clean, mask


def estimate_gravity(clean_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gravity_mg = np.median(clean_xyz, axis=0)
    norm = float(np.linalg.norm(gravity_mg))
    if norm < EPS:
        raise ValueError("Gravity direction could not be estimated (near-zero median vector).")
    return gravity_mg, gravity_mg / norm


def project_perpendicular_to_gravity(
    clean_xyz: np.ndarray,
    gravity_mg: np.ndarray,
    gravity_unit: np.ndarray,
) -> np.ndarray:
    dynamic = clean_xyz - gravity_mg
    parallel = np.outer(dynamic @ gravity_unit, gravity_unit)
    return dynamic - parallel


def butter_bandpass_zero_phase(
    xyz: np.ndarray,
    fs_hz: float,
    fmin_hz: float,
    fmax_hz: float,
    order: int,
) -> np.ndarray:
    nyq = 0.5 * fs_hz
    if not (0 < fmin_hz < fmax_hz < nyq):
        raise ValueError(
            f"Invalid respiratory band [{fmin_hz}, {fmax_hz}] Hz for fs={fs_hz:.3f} Hz."
        )
    sos = signal.butter(order, [fmin_hz, fmax_hz], btype="bandpass", fs=fs_hz, output="sos")
    return signal.sosfiltfilt(sos, xyz, axis=0)


def pca_surrogate(filtered_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    centered = filtered_xyz - np.mean(filtered_xyz, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    surrogate = centered @ eigvecs[:, 0]
    total = float(np.sum(eigvals))
    ratio = float(eigvals[0] / total) if total > EPS else 0.0
    return surrogate, eigvals, eigvecs, ratio
