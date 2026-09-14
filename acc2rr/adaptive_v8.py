from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import Config, _jsonable
from .estimators import kalman_track_rr

ADAPTIVE_V8_VERSION = "adaptive_opt_pc12_hard_soft_v1"

# Experimental target-free hysteresis parameters. These are intentionally
# explicit so real-data validation can judge them without hidden tuning.
V8_HARD_ON_THRESHOLD = 0.45
V8_HARD_OFF_THRESHOLD = 0.25
V8_HARD_ON_WINDOWS = 2
V8_HARD_OFF_WINDOWS = 5

# Soft-fusion ambiguity mapping and temporal smoothing. Engagement is allowed to
# react faster than disengagement so a short dip in ambiguity does not abruptly
# return a harmonic-contaminated PC1 observation to the Kalman filter.
V8_SOFT_LOW = 0.20
V8_SOFT_HIGH = 0.55
V8_SOFT_RISE_RATE = 0.45
V8_SOFT_FALL_RATE = 0.12


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def hard_gate_schmitt(
    ambiguity: np.ndarray,
    *,
    on_threshold: float = V8_HARD_ON_THRESHOLD,
    off_threshold: float = V8_HARD_OFF_THRESHOLD,
    on_windows: int = V8_HARD_ON_WINDOWS,
    off_windows: int = V8_HARD_OFF_WINDOWS,
) -> dict[str, np.ndarray]:
    """Causal Schmitt-style gate from classic PC1/v4 to optimized PC1-PC2.

    Activation requires consecutive high-ambiguity windows. Once adaptive, the
    gate only returns to classic after a longer run of clearly low ambiguity.
    This deliberately makes entry selective and exit conservative without using
    posture, target RR, or any experimental label.
    """
    values = np.asarray(ambiguity, dtype=float)
    if on_windows < 1 or off_windows < 1:
        raise ValueError("on_windows and off_windows must be >= 1")
    if not (0.0 <= off_threshold < on_threshold <= 1.0):
        raise ValueError("Expected 0 <= off_threshold < on_threshold <= 1")

    n = len(values)
    adaptive = np.zeros(n, dtype=bool)
    switches = np.zeros(n, dtype=bool)
    high_streaks = np.zeros(n, dtype=int)
    low_streaks = np.zeros(n, dtype=int)

    mode = False
    high_streak = 0
    low_streak = 0
    for index, raw in enumerate(values):
        score = float(np.clip(raw, 0.0, 1.0)) if np.isfinite(raw) else 0.0
        if not mode:
            high_streak = high_streak + 1 if score >= on_threshold else 0
            low_streak = 0
            if high_streak >= on_windows:
                mode = True
                switches[index] = True
                high_streak = 0
        else:
            low_streak = low_streak + 1 if score <= off_threshold else 0
            high_streak = 0
            if low_streak >= off_windows:
                mode = False
                switches[index] = True
                low_streak = 0

        adaptive[index] = mode
        high_streaks[index] = high_streak
        low_streaks[index] = low_streak

    return {
        "adaptive": adaptive,
        "switch": switches,
        "high_streak": high_streaks,
        "low_streak": low_streaks,
    }


def ambiguity_to_desired_alpha(
    ambiguity: np.ndarray,
    *,
    low: float = V8_SOFT_LOW,
    high: float = V8_SOFT_HIGH,
) -> np.ndarray:
    """Map ambiguity to [0,1] with a smoothstep transition."""
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Expected 0 <= low < high <= 1")
    values = np.asarray(ambiguity, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    x = np.clip((values - low) / (high - low), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smooth_fusion_alpha(
    desired_alpha: np.ndarray,
    *,
    rise_rate: float = V8_SOFT_RISE_RATE,
    fall_rate: float = V8_SOFT_FALL_RATE,
) -> np.ndarray:
    """Causal asymmetric low-pass filter for the soft-fusion weight."""
    desired = np.asarray(desired_alpha, dtype=float)
    if not (0.0 < rise_rate <= 1.0 and 0.0 < fall_rate <= 1.0):
        raise ValueError("rise_rate and fall_rate must lie in (0,1]")
    if not len(desired):
        return np.array([], dtype=float)

    output = np.zeros(len(desired), dtype=float)
    previous = 0.0
    for index, raw in enumerate(desired):
        target = float(np.clip(raw, 0.0, 1.0)) if np.isfinite(raw) else 0.0
        rate = rise_rate if target > previous else fall_rate
        previous = previous + rate * (target - previous)
        output[index] = float(np.clip(previous, 0.0, 1.0))
    return output


def fuse_observations(
    classic_rr: np.ndarray,
    adaptive_rr: np.ndarray,
    classic_confidence: np.ndarray,
    adaptive_confidence: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend classic and optimized-PC12 observations before one Kalman filter."""
    classic_rr = np.asarray(classic_rr, dtype=float)
    adaptive_rr = np.asarray(adaptive_rr, dtype=float)
    classic_confidence = np.asarray(classic_confidence, dtype=float)
    adaptive_confidence = np.asarray(adaptive_confidence, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    n = len(classic_rr)
    if not all(len(values) == n for values in (adaptive_rr, classic_confidence, adaptive_confidence, alpha)):
        raise ValueError("All fusion arrays must have the same length")

    a = np.clip(np.where(np.isfinite(alpha), alpha, 0.0), 0.0, 1.0)
    rr = (1.0 - a) * classic_rr + a * adaptive_rr
    confidence = (1.0 - a) * classic_confidence + a * adaptive_confidence

    # Conservative finite-value fallbacks for diagnostics with isolated missing
    # observations. They do not use any target information.
    classic_finite = np.isfinite(classic_rr)
    adaptive_finite = np.isfinite(adaptive_rr)
    rr = np.where(~classic_finite & adaptive_finite, adaptive_rr, rr)
    rr = np.where(classic_finite & ~adaptive_finite, classic_rr, rr)
    confidence = np.where(
        ~np.isfinite(confidence),
        np.where(a >= 0.5, adaptive_confidence, classic_confidence),
        confidence,
    )
    return rr.astype(float), np.clip(confidence.astype(float), 0.0, 1.0)


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def _error_stats(values: np.ndarray, target: float) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite) or not np.isfinite(target):
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    error = finite - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def _store_error_stats(
    ws: dict[str, Any],
    prefix: str,
    values: np.ndarray,
    target: float,
) -> None:
    stats = _error_stats(values, target)
    ws[f"{prefix}_mae_bpm"] = stats["mae"]
    ws[f"{prefix}_rmse_bpm"] = stats["rmse"]
    ws[f"{prefix}_bias_bpm"] = stats["bias"]


def apply_adaptive_v8(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Run the target-free four-arm v8 experiment after v7.

    Arm A: classic v4 + existing Kalman.
    Arm B: independent optimized_pc12 + one Kalman.
    Arm C: hard Schmitt gate classic-v4 <-> optimized_pc12, then one Kalman.
    Arm D: continuous ambiguity-weighted fusion before one Kalman.

    The experimental target is read only after all observations, gates and
    fusion weights have been selected, solely to compute diagnostic errors.
    """
    del acc_path
    metrics_path = output_dir / "metrics.json"
    windows_path = output_dir / "windowed_rr.csv"
    benchmark_path = output_dir / "surrogate_v5.csv"
    if not (metrics_path.exists() and windows_path.exists() and benchmark_path.exists()):
        raise FileNotFoundError(
            "Adaptive v8 requires metrics.json, windowed_rr.csv and surrogate_v5.csv after v7."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path)
    benchmark = pd.read_csv(benchmark_path)
    if windows.empty:
        metrics["adaptive_v8"] = {
            "version": ADAPTIVE_V8_VERSION,
            "applied": False,
            "reason": "no sliding windows",
        }
        metrics_path.write_text(
            json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return metrics

    required = (
        "rr_temporal_bpm_v4",
        "rr_smoothed_bpm_v4",
        "rr_temporal_bpm",
        "rr_smoothed_bpm",
        "confidence",
        "v7_ambiguity_score",
    )
    missing = [name for name in required if name not in windows.columns]
    if missing:
        raise RuntimeError(f"Adaptive v8 missing v7 window columns: {missing}")

    optimized = benchmark[benchmark["surrogate"] == "optimized_pc12"].sort_values("window_index")
    if len(optimized) != len(windows):
        raise RuntimeError(
            f"Adaptive v8 expected one optimized_pc12 row per window; "
            f"got optimized={len(optimized)}, windows={len(windows)}."
        )
    if not np.array_equal(optimized["window_index"].to_numpy(int), np.arange(len(windows))):
        raise RuntimeError("Adaptive v8 optimized_pc12 rows are not aligned to window_index.")

    # Preserve v7 before the generic temporal/smoothed columns become v8 arm D.
    windows["rr_temporal_bpm_v7"] = windows["rr_temporal_bpm"]
    windows["rr_smoothed_bpm_v7"] = windows["rr_smoothed_bpm"]

    classic_rr = windows["rr_temporal_bpm_v4"].to_numpy(float)
    classic_kalman = windows["rr_smoothed_bpm_v4"].to_numpy(float)
    classic_confidence = windows["confidence"].to_numpy(float)
    adaptive_rr = optimized["rr_consensus_bpm"].to_numpy(float)
    adaptive_confidence = optimized["benchmark_confidence"].to_numpy(float)
    ambiguity = windows["v7_ambiguity_score"].to_numpy(float)

    # Arm B recomputes optimized_pc12 + the same Kalman to keep the v8
    # comparison self-contained and auditable.
    adaptive_kalman = kalman_track_rr(
        adaptive_rr, adaptive_confidence, cfg.hop_s, cfg
    )

    # Arm C: hard asymmetric Schmitt gate, using optimized_pc12 directly rather
    # than the angularly regularized v6 observation.
    hard = hard_gate_schmitt(ambiguity)
    hard_active = hard["adaptive"]
    hard_rr = np.where(hard_active, adaptive_rr, classic_rr)
    hard_confidence = np.where(hard_active, adaptive_confidence, classic_confidence)
    hard_kalman = kalman_track_rr(hard_rr, hard_confidence, cfg.hop_s, cfg)

    # Arm D: continuous target-free fusion. The ambiguity score is first mapped
    # to a smooth weight, then temporally regularized with faster engagement and
    # slower release before one final Kalman filter.
    desired_alpha = ambiguity_to_desired_alpha(ambiguity)
    alpha = smooth_fusion_alpha(desired_alpha)
    soft_rr, soft_confidence = fuse_observations(
        classic_rr,
        adaptive_rr,
        classic_confidence,
        adaptive_confidence,
        alpha,
    )
    soft_kalman = kalman_track_rr(soft_rr, soft_confidence, cfg.hop_s, cfg)

    # Four explicit arms remain available side-by-side.
    windows["rr_v8_classic_bpm"] = classic_rr
    windows["rr_v8_classic_kalman_bpm"] = classic_kalman
    windows["rr_v8_opt_pc12_bpm"] = adaptive_rr
    windows["rr_v8_opt_pc12_kalman_bpm"] = adaptive_kalman

    windows["rr_v8_hard_bpm"] = hard_rr
    windows["rr_v8_hard_kalman_bpm"] = hard_kalman
    windows["v8_hard_mode"] = np.where(hard_active, "adaptive_opt_pc12", "classic_v4")
    windows["v8_hard_adaptive_active"] = hard_active
    windows["v8_hard_mode_switch"] = hard["switch"]
    windows["v8_hard_high_streak"] = hard["high_streak"]
    windows["v8_hard_low_streak"] = hard["low_streak"]
    windows["v8_hard_selected_confidence"] = hard_confidence

    windows["v8_soft_desired_alpha"] = desired_alpha
    windows["v8_soft_alpha"] = alpha
    windows["rr_v8_soft_bpm"] = soft_rr
    windows["rr_v8_soft_kalman_bpm"] = soft_kalman
    windows["v8_soft_selected_confidence"] = soft_confidence

    # Arm D is promoted to the generic v8 output only for the experiment. All
    # earlier versions remain explicitly preserved above.
    windows["rr_temporal_bpm"] = soft_rr
    windows["rr_smoothed_bpm"] = soft_kalman

    target_rpm = _finite(metrics.get("target_rpm"))
    if np.isfinite(target_rpm):
        windows["error_temporal_bpm"] = soft_rr - target_rpm
        windows["error_smoothed_bpm"] = soft_kalman - target_rpm

    windows.to_csv(windows_path, index=False)

    diagnostic_columns = [
        "time_s",
        "v7_ambiguity_score",
        "rr_v8_classic_bpm",
        "rr_v8_classic_kalman_bpm",
        "rr_v8_opt_pc12_bpm",
        "rr_v8_opt_pc12_kalman_bpm",
        "rr_v8_hard_bpm",
        "rr_v8_hard_kalman_bpm",
        "v8_hard_mode",
        "v8_hard_adaptive_active",
        "v8_hard_mode_switch",
        "v8_hard_high_streak",
        "v8_hard_low_streak",
        "v8_hard_selected_confidence",
        "v8_soft_desired_alpha",
        "v8_soft_alpha",
        "rr_v8_soft_bpm",
        "rr_v8_soft_kalman_bpm",
        "v8_soft_selected_confidence",
        "rr_temporal_bpm_v6",
        "rr_smoothed_bpm_v6",
        "rr_temporal_bpm_v7",
        "rr_smoothed_bpm_v7",
    ]
    diagnostic_columns = [name for name in diagnostic_columns if name in windows.columns]
    windows[diagnostic_columns].to_csv(output_dir / "adaptive_v8.csv", index=False)

    ws = dict(metrics.get("window_stats", {}))
    # Preserve generic v7 statistics before arm D becomes the generic v8 output.
    for key in (
        "rr_temporal_mean_bpm",
        "rr_temporal_median_bpm",
        "rr_temporal_std_bpm",
        "rr_smoothed_mean_bpm",
        "rr_smoothed_median_bpm",
        "rr_smoothed_std_bpm",
        "mae_temporal_bpm",
        "rmse_temporal_bpm",
        "bias_temporal_bpm",
        "mae_smoothed_bpm",
        "rmse_smoothed_bpm",
        "bias_smoothed_bpm",
    ):
        if key in ws and f"v7_{key}" not in ws:
            ws[f"v7_{key}"] = ws[key]

    valid_mask = (
        windows["valid"].to_numpy(bool)
        if "valid" in windows.columns
        else np.ones(len(windows), dtype=bool)
    )
    temporal_valid = soft_rr[valid_mask]
    smoothed_valid = soft_kalman[valid_mask]
    temporal_stats = _stats(temporal_valid)
    smoothed_stats = _stats(smoothed_valid)
    ws.update(
        {
            "rr_temporal_mean_bpm": temporal_stats["mean"],
            "rr_temporal_median_bpm": temporal_stats["median"],
            "rr_temporal_std_bpm": temporal_stats["std"],
            "rr_smoothed_mean_bpm": smoothed_stats["mean"],
            "rr_smoothed_median_bpm": smoothed_stats["median"],
            "rr_smoothed_std_bpm": smoothed_stats["std"],
            "v8_hard_adaptive_fraction": float(np.mean(hard_active)),
            "v8_hard_mode_switch_count": int(np.sum(hard["switch"])),
            "v8_soft_alpha_mean": float(np.mean(alpha)),
            "v8_soft_alpha_median": float(np.median(alpha)),
            "v8_soft_alpha_p90": float(np.percentile(alpha, 90)),
            "v8_soft_alpha_ge_0_5_fraction": float(np.mean(alpha >= 0.5)),
        }
    )

    if np.isfinite(target_rpm):
        # Generic metrics remain the v8 soft arm on the legacy valid mask.
        valid_temporal_error = _error_stats(temporal_valid, target_rpm)
        valid_smoothed_error = _error_stats(smoothed_valid, target_rpm)
        ws.update(
            {
                "mae_temporal_bpm": valid_temporal_error["mae"],
                "rmse_temporal_bpm": valid_temporal_error["rmse"],
                "bias_temporal_bpm": valid_temporal_error["bias"],
                "mae_smoothed_bpm": valid_smoothed_error["mae"],
                "rmse_smoothed_bpm": valid_smoothed_error["rmse"],
                "bias_smoothed_bpm": valid_smoothed_error["bias"],
            }
        )
        for prefix, values in (
            ("v8_arm_a_classic_all", classic_rr),
            ("v8_arm_a_classic_kalman_all", classic_kalman),
            ("v8_arm_b_opt_all", adaptive_rr),
            ("v8_arm_b_opt_kalman_all", adaptive_kalman),
            ("v8_arm_c_hard_all", hard_rr),
            ("v8_arm_c_hard_kalman_all", hard_kalman),
            ("v8_arm_d_soft_all", soft_rr),
            ("v8_arm_d_soft_kalman_all", soft_kalman),
        ):
            _store_error_stats(ws, prefix, values, target_rpm)

    metrics["window_stats"] = ws
    metrics["adaptive_v8"] = {
        "version": ADAPTIVE_V8_VERSION,
        "applied": True,
        "selection_uses_target_rpm": False,
        "arm_a": "classic v4 observation + existing Kalman",
        "arm_b": "independent optimized_pc12 + one Kalman",
        "arm_c": "hard Schmitt gate classic_v4 <-> optimized_pc12, then one Kalman",
        "arm_d": "continuous ambiguity-weighted classic/optimized fusion, then one Kalman",
        "generic_output": "arm_d_soft_fusion",
        "ambiguity_source": "v7_ambiguity_score (target-free)",
        "hard_on_threshold": V8_HARD_ON_THRESHOLD,
        "hard_off_threshold": V8_HARD_OFF_THRESHOLD,
        "hard_on_windows": V8_HARD_ON_WINDOWS,
        "hard_off_windows": V8_HARD_OFF_WINDOWS,
        "soft_low": V8_SOFT_LOW,
        "soft_high": V8_SOFT_HIGH,
        "soft_rise_rate": V8_SOFT_RISE_RATE,
        "soft_fall_rate": V8_SOFT_FALL_RATE,
        "hard_adaptive_fraction": float(np.mean(hard_active)),
        "hard_mode_switch_count": int(np.sum(hard["switch"])),
        "soft_alpha_mean": float(np.mean(alpha)),
        "soft_alpha_median": float(np.median(alpha)),
    }
    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _plot_adaptive_v8(output_dir / "adaptive_v8.png", windows, target_rpm, cfg)
    return metrics


def _plot_adaptive_v8(
    path: Path,
    windows: pd.DataFrame,
    target_rpm: float,
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    time_s = windows["time_s"]

    axes[0].plot(time_s, windows["rr_v8_classic_kalman_bpm"], label="A classic v4 + Kalman")
    axes[0].plot(time_s, windows["rr_v8_opt_pc12_kalman_bpm"], label="B opt-PC12 + Kalman")
    axes[0].plot(time_s, windows["rr_v8_hard_kalman_bpm"], label="C hard gate + Kalman")
    axes[0].plot(time_s, windows["rr_v8_soft_kalman_bpm"], label="D soft fusion + Kalman")
    if np.isfinite(target_rpm):
        axes[0].axhline(target_rpm, linestyle="--", label=f"Alvo {target_rpm:g}")
    axes[0].set_ylabel("RR [resp/min]")
    axes[0].set_title(f"Adaptive v8 — quatro braços ({cfg.window_s:g}s / hop {cfg.hop_s:g}s)")
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(time_s, windows["v7_ambiguity_score"], label="Ambiguidade")
    axes[1].plot(time_s, windows["v8_soft_desired_alpha"], label="Alpha desejado", alpha=0.65)
    axes[1].plot(time_s, windows["v8_soft_alpha"], label="Alpha soft efetivo")
    axes[1].axhline(V8_HARD_ON_THRESHOLD, linestyle="--", label="Hard ON")
    axes[1].axhline(V8_HARD_OFF_THRESHOLD, linestyle=":", label="Hard OFF")
    hard_numeric = windows["v8_hard_adaptive_active"].astype(float)
    axes[1].fill_between(time_s, 0, hard_numeric, alpha=0.12, label="Hard adaptive ativo")
    axes[1].set_ylabel("Score / alpha")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_title("Gate Schmitt e soft fusion")
    axes[1].legend(loc="upper right", ncols=2)

    axes[2].plot(time_s, windows["confidence"], label="Confiança clássica")
    axes[2].plot(time_s, windows["v7_opt_pc12_confidence"], label="Confiança opt-PC12")
    axes[2].plot(time_s, windows["v8_hard_selected_confidence"], label="Confiança hard")
    axes[2].plot(time_s, windows["v8_soft_selected_confidence"], label="Confiança soft")
    axes[2].set_xlabel("Tempo [s]")
    axes[2].set_ylabel("Confiança")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Confiança das observações antes do Kalman")
    axes[2].legend(loc="upper right", ncols=2)

    fig.savefig(path, dpi=160)
    plt.close(fig)
