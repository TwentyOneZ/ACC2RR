from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import Config, _jsonable
from .estimators import kalman_track_rr
from .temporal import HARMONIC_RATIO_TOLERANCE

ADAPTIVE_V7_VERSION = "adaptive_three_arm_v1"
V7_MODE_SWITCH_PENALTY = 0.35
V7_INITIAL_ADAPTIVE_PENALTY = 0.20
V7_INELIGIBLE_ADAPTIVE_PENALTY = 0.60


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _harmonic_disagreement(rr_psd_bpm: float, rr_acf_bpm: float) -> bool:
    if not (
        np.isfinite(rr_psd_bpm)
        and np.isfinite(rr_acf_bpm)
        and rr_psd_bpm > 0
        and rr_acf_bpm > 0
    ):
        return False
    ratio = max(rr_psd_bpm, rr_acf_bpm) / min(rr_psd_bpm, rr_acf_bpm)
    multiple = min((2, 3), key=lambda m: abs(ratio - m))
    return abs(ratio - multiple) / multiple <= HARMONIC_RATIO_TOLERANCE


def compute_ambiguity_evidence(
    pc1: dict[str, Any] | pd.Series,
    optimized: dict[str, Any] | pd.Series,
) -> dict[str, float | bool]:
    """Compute a target-free need-for-adaptation score for one window.

    The gate deliberately combines several weak clues instead of hard-coding a
    posture or a single PCA-variance threshold. The full-record ACC estimate may
    appear indirectly through reference_fundamental_to_2x_ratio, but the
    experimental target never participates in this function.
    """
    rr_psd = _finite(pc1.get("rr_psd_bpm"))
    rr_acf = _finite(pc1.get("rr_acf_bpm"))
    harmonic = _harmonic_disagreement(rr_psd, rr_acf)

    reference_ratio = _finite(pc1.get("reference_fundamental_to_2x_ratio"))
    if np.isfinite(reference_ratio):
        ratio_evidence = float(np.clip((0.60 - reference_ratio) / 0.50, 0.0, 1.0))
    else:
        ratio_evidence = 0.0

    pc1_variance = _finite(pc1.get("pca_explained_variance_ratio"))
    if np.isfinite(pc1_variance):
        variance_evidence = float(np.clip((0.88 - pc1_variance) / 0.20, 0.0, 1.0))
    else:
        variance_evidence = 0.0

    pc1_confidence = _finite(pc1.get("benchmark_confidence"), 0.0)
    optimized_confidence = _finite(optimized.get("benchmark_confidence"), 0.0)
    confidence_gain = optimized_confidence - pc1_confidence
    gain_evidence = float(np.clip((confidence_gain - 0.02) / 0.15, 0.0, 1.0))

    pc1_agreement = _finite(pc1.get("agreement_quality"), 0.0)
    agreement_deficit = float(np.clip((0.75 - pc1_agreement) / 0.45, 0.0, 1.0))

    ambiguity = float(
        np.clip(
            0.30 * float(harmonic)
            + 0.25 * ratio_evidence
            + 0.20 * variance_evidence
            + 0.20 * gain_evidence
            + 0.05 * agreement_deficit,
            0.0,
            1.0,
        )
    )

    eligible = bool(
        (harmonic and (ratio_evidence >= 0.35 or gain_evidence >= 0.35))
        or (ratio_evidence >= 0.55 and gain_evidence >= 0.35)
        or (variance_evidence >= 0.65 and gain_evidence >= 0.45)
    )

    classic_score = float(
        np.clip((1.0 - ambiguity) * (0.65 + 0.35 * pc1_confidence), 0.0, 1.0)
    )
    adaptive_score = float(
        np.clip(ambiguity * (0.65 + 0.35 * optimized_confidence), 0.0, 1.0)
    )

    return {
        "ambiguity_score": ambiguity,
        "harmonic_disagreement": harmonic,
        "reference_ratio_evidence": ratio_evidence,
        "variance_evidence": variance_evidence,
        "confidence_gain": float(confidence_gain),
        "confidence_gain_evidence": gain_evidence,
        "agreement_deficit": agreement_deficit,
        "adaptive_eligible": eligible,
        "classic_score": classic_score,
        "adaptive_score": adaptive_score,
    }


def select_hybrid_modes(
    classic_scores: np.ndarray,
    adaptive_scores: np.ndarray,
    adaptive_eligible: np.ndarray,
) -> dict[str, np.ndarray]:
    """Two-state Viterbi hysteresis for classic-v4 vs adaptive-v6 operation."""
    classic_scores = np.asarray(classic_scores, dtype=float)
    adaptive_scores = np.asarray(adaptive_scores, dtype=float)
    adaptive_eligible = np.asarray(adaptive_eligible, dtype=bool)
    n = len(classic_scores)
    if not (len(adaptive_scores) == n and len(adaptive_eligible) == n):
        raise ValueError("Mode evidence arrays must have identical length.")
    if n == 0:
        empty = np.array([], dtype=float)
        return {
            "mode": np.array([], dtype=int),
            "selected_score": empty,
            "switch_cost": empty,
        }

    # mode 0 = classic v4; mode 1 = adaptive v6.
    costs = np.full((n, 2), np.inf, dtype=float)
    back = np.full((n, 2), -1, dtype=int)
    switch_costs = np.zeros((n, 2), dtype=float)

    costs[0, 0] = -classic_scores[0]
    costs[0, 1] = (
        -adaptive_scores[0]
        + V7_INITIAL_ADAPTIVE_PENALTY
        + (0.0 if adaptive_eligible[0] else V7_INELIGIBLE_ADAPTIVE_PENALTY)
    )

    for index in range(1, n):
        emissions = np.array(
            [
                -classic_scores[index],
                -adaptive_scores[index]
                + (0.0 if adaptive_eligible[index] else V7_INELIGIBLE_ADAPTIVE_PENALTY),
            ],
            dtype=float,
        )
        for current_mode in (0, 1):
            for previous_mode in (0, 1):
                switch = V7_MODE_SWITCH_PENALTY if current_mode != previous_mode else 0.0
                total = costs[index - 1, previous_mode] + switch + emissions[current_mode]
                if total < costs[index, current_mode]:
                    costs[index, current_mode] = total
                    back[index, current_mode] = previous_mode
                    switch_costs[index, current_mode] = switch

    mode = np.zeros(n, dtype=int)
    mode[-1] = int(np.argmin(costs[-1]))
    for index in range(n - 1, 0, -1):
        mode[index - 1] = back[index, mode[index]]

    selected_score = np.where(mode == 0, classic_scores, adaptive_scores)
    selected_switch = np.array(
        [switch_costs[index, mode[index]] for index in range(n)],
        dtype=float,
    )
    return {
        "mode": mode,
        "selected_score": selected_score,
        "switch_cost": selected_switch,
    }


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


def apply_adaptive_v7(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Run the target-free A/B/C v7 experiment after v6.

    Arm A: independent optimized_pc12 + Kalman.
    Arm B: joint v6 + Kalman (preserved exactly).
    Arm C: target-free hybrid choosing classic v4 or adaptive v6 with hysteresis.
    """
    del acc_path  # The required signal diagnostics have already been persisted by v5/v6.
    metrics_path = output_dir / "metrics.json"
    windows_path = output_dir / "windowed_rr.csv"
    benchmark_path = output_dir / "surrogate_v5.csv"
    if not (metrics_path.exists() and windows_path.exists() and benchmark_path.exists()):
        raise FileNotFoundError(
            "Adaptive v7 requires metrics.json, windowed_rr.csv and surrogate_v5.csv after v6."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path)
    benchmark = pd.read_csv(benchmark_path)
    if windows.empty:
        metrics["adaptive_v7"] = {
            "version": ADAPTIVE_V7_VERSION,
            "applied": False,
            "reason": "no sliding windows",
        }
        metrics_path.write_text(
            json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return metrics

    required_window_columns = (
        "rr_temporal_bpm_v4",
        "rr_smoothed_bpm_v4",
        "rr_temporal_bpm",
        "rr_smoothed_bpm",
        "v6_benchmark_confidence",
        "confidence",
    )
    missing = [name for name in required_window_columns if name not in windows.columns]
    if missing:
        raise RuntimeError(f"Adaptive v7 missing v6 window columns: {missing}")

    pc1 = benchmark[benchmark["surrogate"] == "pc1"].sort_values("window_index")
    optimized = benchmark[benchmark["surrogate"] == "optimized_pc12"].sort_values("window_index")
    if len(pc1) != len(windows) or len(optimized) != len(windows):
        raise RuntimeError(
            f"Adaptive v7 expected one pc1 and optimized_pc12 row per window; "
            f"got pc1={len(pc1)}, optimized={len(optimized)}, windows={len(windows)}."
        )
    if not (
        np.array_equal(pc1["window_index"].to_numpy(int), np.arange(len(windows)))
        and np.array_equal(optimized["window_index"].to_numpy(int), np.arange(len(windows)))
    ):
        raise RuntimeError("Adaptive v7 surrogate rows are not aligned to window_index.")

    # Preserve arm B (v6) before generic columns become arm C (hybrid v7).
    windows["rr_temporal_bpm_v6"] = windows["rr_temporal_bpm"]
    windows["rr_smoothed_bpm_v6"] = windows["rr_smoothed_bpm"]

    # Arm A: independent v5 winner per window followed by exactly the same Kalman.
    arm_a_rr = optimized["rr_consensus_bpm"].to_numpy(float)
    arm_a_confidence = optimized["benchmark_confidence"].to_numpy(float)
    arm_a_kalman = kalman_track_rr(arm_a_rr, arm_a_confidence, cfg.hop_s, cfg)
    windows["rr_v5_opt_pc12_bpm"] = arm_a_rr
    windows["rr_v5_opt_pc12_kalman_bpm"] = arm_a_kalman
    windows["v7_opt_pc12_confidence"] = arm_a_confidence

    evidence_rows: list[dict[str, float | bool]] = []
    for index in range(len(windows)):
        evidence_rows.append(
            compute_ambiguity_evidence(pc1.iloc[index], optimized.iloc[index])
        )
    evidence = pd.DataFrame(evidence_rows)

    mode_track = select_hybrid_modes(
        evidence["classic_score"].to_numpy(float),
        evidence["adaptive_score"].to_numpy(float),
        evidence["adaptive_eligible"].to_numpy(bool),
    )
    mode = mode_track["mode"]

    v4_rr = windows["rr_temporal_bpm_v4"].to_numpy(float)
    v6_rr = windows["rr_temporal_bpm_v6"].to_numpy(float)
    v4_confidence = windows["confidence"].to_numpy(float)
    v6_confidence = windows["v6_benchmark_confidence"].to_numpy(float)

    hybrid_rr = np.where(mode == 0, v4_rr, v6_rr)
    hybrid_confidence = np.where(mode == 0, v4_confidence, v6_confidence)
    hybrid_kalman = kalman_track_rr(hybrid_rr, hybrid_confidence, cfg.hop_s, cfg)

    windows["rr_temporal_bpm"] = hybrid_rr
    windows["rr_smoothed_bpm"] = hybrid_kalman
    windows["v7_mode"] = np.where(mode == 0, "classic_v4", "adaptive_v6")
    windows["v7_adaptive_active"] = mode == 1
    windows["v7_mode_switch"] = np.r_[False, mode[1:] != mode[:-1]]
    windows["v7_mode_switch_cost"] = mode_track["switch_cost"]
    windows["v7_selected_mode_score"] = mode_track["selected_score"]
    windows["v7_selected_confidence"] = hybrid_confidence

    for column in evidence.columns:
        windows[f"v7_{column}"] = evidence[column].to_numpy()

    target_rpm = _finite(metrics.get("target_rpm"))
    if np.isfinite(target_rpm):
        windows["error_temporal_bpm"] = windows["rr_temporal_bpm"] - target_rpm
        windows["error_smoothed_bpm"] = windows["rr_smoothed_bpm"] - target_rpm

    windows.to_csv(windows_path, index=False)

    diagnostic_columns = [
        "time_s",
        "rr_v5_opt_pc12_bpm",
        "rr_v5_opt_pc12_kalman_bpm",
        "rr_temporal_bpm_v4",
        "rr_smoothed_bpm_v4",
        "rr_temporal_bpm_v6",
        "rr_smoothed_bpm_v6",
        "rr_temporal_bpm",
        "rr_smoothed_bpm",
        "v7_mode",
        "v7_adaptive_active",
        "v7_mode_switch",
        "v7_ambiguity_score",
        "v7_harmonic_disagreement",
        "v7_reference_ratio_evidence",
        "v7_variance_evidence",
        "v7_confidence_gain",
        "v7_confidence_gain_evidence",
        "v7_agreement_deficit",
        "v7_adaptive_eligible",
        "v7_classic_score",
        "v7_adaptive_score",
        "v7_selected_confidence",
        "v7_opt_pc12_confidence",
    ]
    windows[diagnostic_columns].to_csv(output_dir / "adaptive_v7.csv", index=False)

    ws = dict(metrics.get("window_stats", {}))
    # Preserve the generic v6 statistics before arm C becomes current.
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
        if key in ws and f"v6_{key}" not in ws:
            ws[f"v6_{key}"] = ws[key]

    valid_mask = (
        windows["valid"].to_numpy(bool)
        if "valid" in windows.columns
        else np.ones(len(windows), dtype=bool)
    )
    hybrid_stats = _stats(hybrid_rr[valid_mask])
    hybrid_kalman_stats = _stats(hybrid_kalman[valid_mask])
    ws.update(
        {
            "rr_temporal_mean_bpm": hybrid_stats["mean"],
            "rr_temporal_median_bpm": hybrid_stats["median"],
            "rr_temporal_std_bpm": hybrid_stats["std"],
            "rr_smoothed_mean_bpm": hybrid_kalman_stats["mean"],
            "rr_smoothed_median_bpm": hybrid_kalman_stats["median"],
            "rr_smoothed_std_bpm": hybrid_kalman_stats["std"],
            "v7_adaptive_fraction": float(np.mean(mode == 1)),
            "v7_mode_switch_count": int(np.sum(mode[1:] != mode[:-1])),
            "v7_ambiguity_mean": float(np.mean(evidence["ambiguity_score"])),
            "v7_ambiguity_median": float(np.median(evidence["ambiguity_score"])),
            "v7_eligible_fraction": float(np.mean(evidence["adaptive_eligible"])),
            "v7_arm_a_confidence_mean": float(np.nanmean(arm_a_confidence)),
            "v7_arm_c_confidence_mean": float(np.nanmean(hybrid_confidence)),
        }
    )

    if np.isfinite(target_rpm):
        arms = {
            "arm_a_opt": arm_a_rr,
            "arm_a_opt_kalman": arm_a_kalman,
            "arm_b_v6": v6_rr,
            "arm_b_v6_kalman": windows["rr_smoothed_bpm_v6"].to_numpy(float),
            "arm_c_hybrid": hybrid_rr,
            "arm_c_hybrid_kalman": hybrid_kalman,
        }
        for name, values in arms.items():
            all_error = _error_stats(values, target_rpm)
            valid_error = _error_stats(values[valid_mask], target_rpm)
            ws[f"v7_{name}_all_mae_bpm"] = all_error["mae"]
            ws[f"v7_{name}_all_rmse_bpm"] = all_error["rmse"]
            ws[f"v7_{name}_all_bias_bpm"] = all_error["bias"]
            ws[f"v7_{name}_valid_mae_bpm"] = valid_error["mae"]
            ws[f"v7_{name}_valid_rmse_bpm"] = valid_error["rmse"]
            ws[f"v7_{name}_valid_bias_bpm"] = valid_error["bias"]

        current_temporal = _error_stats(hybrid_rr[valid_mask], target_rpm)
        current_smoothed = _error_stats(hybrid_kalman[valid_mask], target_rpm)
        ws.update(
            {
                "mae_temporal_bpm": current_temporal["mae"],
                "rmse_temporal_bpm": current_temporal["rmse"],
                "bias_temporal_bpm": current_temporal["bias"],
                "mae_smoothed_bpm": current_smoothed["mae"],
                "rmse_smoothed_bpm": current_smoothed["rmse"],
                "bias_smoothed_bpm": current_smoothed["bias"],
            }
        )

    metrics["window_stats"] = ws
    metrics["adaptive_v7"] = {
        "version": ADAPTIVE_V7_VERSION,
        "applied": True,
        "selection_uses_target_rpm": False,
        "arm_a": "independent optimized_pc12 + Kalman",
        "arm_b": "joint RR-orientation v6 + Kalman",
        "arm_c": "target-free hybrid classic_v4/adaptive_v6 + Kalman",
        "classic_source": "rr_temporal_bpm_v4",
        "adaptive_source": "rr_temporal_bpm_v6",
        "ambiguity_inputs": [
            "PC1 PSD-ACF integer-harmonic disagreement",
            "PC1 full-record-reference fundamental/2x ratio",
            "PC1 explained variance ratio",
            "optimized_pc12 minus PC1 benchmark-confidence gain",
            "PC1 PSD-ACF agreement deficit",
        ],
        "mode_switch_penalty": V7_MODE_SWITCH_PENALTY,
        "initial_adaptive_penalty": V7_INITIAL_ADAPTIVE_PENALTY,
        "ineligible_adaptive_penalty": V7_INELIGIBLE_ADAPTIVE_PENALTY,
        "adaptive_fraction": float(np.mean(mode == 1)),
        "mode_switch_count": int(np.sum(mode[1:] != mode[:-1])),
        "ambiguity_mean": float(np.mean(evidence["ambiguity_score"])),
        "eligible_fraction": float(np.mean(evidence["adaptive_eligible"])),
    }
    metrics["temporal_tracker"] = metrics["adaptive_v7"]
    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    full_est = metrics.get("full_record_estimate", {})
    global_rr = _finite(full_est.get("rr_final_bpm"))
    _plot_v7(output_dir / "adaptive_v7.png", windows, target_rpm, global_rr)
    return metrics


def _plot_v7(path: Path, windows: pd.DataFrame, target_rpm: float, global_rr: float) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    time_s = windows["time_s"]

    axes[0].plot(time_s, windows["rr_smoothed_bpm_v4"], label="Classic v4 + Kalman", alpha=0.65)
    axes[0].plot(time_s, windows["rr_v5_opt_pc12_kalman_bpm"], label="Arm A: opt PC12 + Kalman", alpha=0.75)
    axes[0].plot(time_s, windows["rr_smoothed_bpm_v6"], label="Arm B: v6 + Kalman", alpha=0.75)
    axes[0].plot(time_s, windows["rr_smoothed_bpm"], label="Arm C: hybrid v7 + Kalman", linewidth=2.0)
    if np.isfinite(global_rr):
        axes[0].axhline(global_rr, linestyle=":", label=f"Full-record RR {global_rr:.2f}")
    if np.isfinite(target_rpm):
        axes[0].axhline(target_rpm, linestyle="--", label=f"Experimental target {target_rpm:g}")
    axes[0].set_ylabel("RR [resp/min]")
    axes[0].set_title("Adaptive v7 — A/B/C comparison")
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(time_s, windows["v7_ambiguity_score"], label="Ambiguity score")
    axes[1].plot(time_s, windows["v7_classic_score"], label="Classic score", alpha=0.75)
    axes[1].plot(time_s, windows["v7_adaptive_score"], label="Adaptive score", alpha=0.75)
    active = windows["v7_adaptive_active"].to_numpy(bool)
    if active.any():
        axes[1].scatter(time_s[active], windows.loc[active, "v7_ambiguity_score"], marker=".", label="Adaptive mode")
    axes[1].set_ylabel("Target-free evidence")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Hybrid gate evidence and selected mode")
    axes[1].legend(loc="upper right", ncols=2)

    axes[2].plot(time_s, windows["v7_opt_pc12_confidence"], label="Opt-PC12 confidence")
    axes[2].plot(time_s, windows["v6_benchmark_confidence"], label="v6 confidence")
    axes[2].plot(time_s, windows["confidence"], label="Classic PC1 confidence", alpha=0.75)
    axes[2].plot(time_s, windows["v7_selected_confidence"], label="Hybrid selected confidence", linewidth=2.0)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Confidence")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Observation confidence by arm")
    axes[2].legend(loc="upper right", ncols=2)

    fig.savefig(path, dpi=160)
    plt.close(fig)
