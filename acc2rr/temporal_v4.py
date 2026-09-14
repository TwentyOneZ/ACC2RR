from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    Config,
    _finite_float,
    _jsonable,
    butter_bandpass_zero_phase,
    estimate_gravity,
    hampel_vector_despike,
    load_metadata,
    pca_surrogate,
    project_perpendicular_to_gravity,
    read_acc_csv,
    resample_uniform,
    timestamp_qc_and_segment,
)
from .estimators import estimate_surrogate, kalman_track_rr, motion_quality
from .temporal import HARMONIC_RATIO_TOLERANCE, temporal_candidate_set

TEMPORAL_TRACKER_V4_VERSION = "harmonic_viterbi_v2_prior_asymmetric"
V4_CONTINUITY_PENALTY_PER_BPM = 0.018
V4_UPWARD_HARMONIC_PENALTY = 6.0
V4_DOWNWARD_HARMONIC_PENALTY_MIN = 1.2
V4_DOWNWARD_HARMONIC_PENALTY_MAX = 3.0
V4_GLOBAL_PRIOR_MAX_COST = 4.5
V4_EMISSION_WEIGHT_FLOOR = 0.25


def _near_integer_harmonic(rr_a: float, rr_b: float) -> tuple[bool, int, float]:
    if not (np.isfinite(rr_a) and np.isfinite(rr_b) and rr_a > 0 and rr_b > 0):
        return False, 1, float("inf")
    ratio = max(rr_a, rr_b) / min(rr_a, rr_b)
    multiple = min((2, 3), key=lambda m: abs(ratio - m))
    rel_error = abs(ratio - multiple) / multiple
    return rel_error <= HARMONIC_RATIO_TOLERANCE, int(multiple), float(rel_error)


def _emission_weight(confidence: float) -> float:
    c = float(np.clip(confidence, 0.0, 1.0)) if np.isfinite(confidence) else 0.0
    return V4_EMISSION_WEIGHT_FLOOR + (1.0 - V4_EMISSION_WEIGHT_FLOOR) * c


def _initial_prior_cost(
    rr_bpm: float,
    global_prior_rr_bpm: float,
    global_prior_confidence: float,
) -> float:
    if not (
        np.isfinite(rr_bpm)
        and rr_bpm > 0
        and np.isfinite(global_prior_rr_bpm)
        and global_prior_rr_bpm > 0
    ):
        return 0.0
    confidence = float(np.clip(global_prior_confidence, 0.0, 1.0))
    scale_bpm = max(2.5, 0.15 * global_prior_rr_bpm)
    normalized = abs(rr_bpm - global_prior_rr_bpm) / scale_bpm
    return float(
        V4_GLOBAL_PRIOR_MAX_COST
        * confidence
        * min(normalized * normalized, 1.0)
    )


def _transition_cost_v4(
    previous_candidate: dict[str, float],
    current_candidate: dict[str, float],
    current_confidence: float,
    hop_s: float,
) -> tuple[float, bool, str]:
    previous_rr = float(previous_candidate["rr_bpm"])
    current_rr = float(current_candidate["rr_bpm"])
    hop_scale = max(float(hop_s), 1.0)

    cost = (
        V4_CONTINUITY_PENALTY_PER_BPM
        * abs(current_rr - previous_rr)
        / hop_scale
    )
    harmonic_jump, multiple, _ = _near_integer_harmonic(previous_rr, current_rr)
    if not harmonic_jump:
        return float(cost), False, ""

    acf_support = float(np.clip(current_candidate.get("acf_support", 0.0), 0.0, 1.0))
    confidence = (
        float(np.clip(current_confidence, 0.0, 1.0))
        if np.isfinite(current_confidence)
        else 0.0
    )

    if current_rr > previous_rr:
        # Moving from a plausible fundamental to 2x/3x is expensive when the
        # higher state has weak ACF support or low local confidence. A genuine
        # sustained rate increase with strong ACF support can still overcome it.
        evidence = float(np.clip((acf_support * confidence) / 0.55, 0.0, 1.0))
        harmonic_penalty = V4_UPWARD_HARMONIC_PENALTY * (1.0 - 0.55 * evidence)
        relation = f"fundamental_to_{multiple}x"
    else:
        # Returning from a harmonic to an ACF-supported lower fundamental is
        # deliberately cheaper. Weak ACF support raises the cost toward the
        # conservative ceiling.
        harmonic_penalty = (
            V4_DOWNWARD_HARMONIC_PENALTY_MIN
            + (
                V4_DOWNWARD_HARMONIC_PENALTY_MAX
                - V4_DOWNWARD_HARMONIC_PENALTY_MIN
            )
            * (1.0 - acf_support)
        )
        relation = f"{multiple}x_to_fundamental"

    cost += harmonic_penalty / hop_scale
    return float(cost), True, relation


def harmonic_viterbi_track_v4(
    candidate_sets: list[list[dict[str, float]]],
    confidence: np.ndarray,
    global_prior_rr_bpm: float,
    global_prior_confidence: float,
    hop_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Track RR using weak full-record initialization and asymmetric harmonics.

    The full-record RR is used only as a soft initial boundary condition. Local
    evidence is confidence-weighted. Integer-harmonic transitions are
    directional: an ACF-supported return from 2f/3f to f is cheaper than an
    unsupported jump from f to its harmonic.
    """
    if not candidate_sets:
        empty = np.array([], dtype=float)
        return {
            "rr_bpm": empty,
            "selected_score": empty,
            "emission_weight": empty,
            "transition_cost": empty,
            "harmonic_transition": np.array([], dtype=bool),
            "transition_relation": np.array([], dtype=object),
            "initial_prior_cost": empty,
            "path_index": np.array([], dtype=int),
        }
    if any(len(candidates) == 0 for candidates in candidate_sets):
        raise ValueError("Every temporal window must contain at least one candidate.")
    if len(confidence) != len(candidate_sets):
        raise ValueError("confidence must contain one value per candidate set.")

    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    transition_costs: list[np.ndarray] = []
    transition_harmonic: list[np.ndarray] = []
    transition_relation: list[np.ndarray] = []
    prior_costs: list[np.ndarray] = []

    first = candidate_sets[0]
    first_weight = _emission_weight(float(confidence[0]))
    first_prior = np.array(
        [
            _initial_prior_cost(
                float(candidate["rr_bpm"]),
                global_prior_rr_bpm,
                global_prior_confidence,
            )
            for candidate in first
        ],
        dtype=float,
    )
    costs.append(
        np.array(
            [
                -float(candidate["score"]) * first_weight + first_prior[j]
                for j, candidate in enumerate(first)
            ],
            dtype=float,
        )
    )
    backpointers.append(np.full(len(first), -1, dtype=int))
    transition_costs.append(np.zeros(len(first), dtype=float))
    transition_harmonic.append(np.zeros(len(first), dtype=bool))
    transition_relation.append(np.full(len(first), "", dtype=object))
    prior_costs.append(first_prior)

    for index in range(1, len(candidate_sets)):
        previous = candidate_sets[index - 1]
        current = candidate_sets[index]
        current_cost = np.full(len(current), np.inf, dtype=float)
        current_back = np.full(len(current), -1, dtype=int)
        current_transition = np.full(len(current), np.nan, dtype=float)
        current_harmonic = np.zeros(len(current), dtype=bool)
        current_relation = np.full(len(current), "", dtype=object)
        current_weight = _emission_weight(float(confidence[index]))

        for j, candidate in enumerate(current):
            emission_cost = -float(candidate["score"]) * current_weight
            for k, prev_candidate in enumerate(previous):
                trans_cost, harmonic_jump, relation = _transition_cost_v4(
                    prev_candidate,
                    candidate,
                    float(confidence[index]),
                    hop_s,
                )
                total = costs[index - 1][k] + trans_cost + emission_cost
                if total < current_cost[j]:
                    current_cost[j] = total
                    current_back[j] = k
                    current_transition[j] = trans_cost
                    current_harmonic[j] = harmonic_jump
                    current_relation[j] = relation

        costs.append(current_cost)
        backpointers.append(current_back)
        transition_costs.append(current_transition)
        transition_harmonic.append(current_harmonic)
        transition_relation.append(current_relation)
        prior_costs.append(np.zeros(len(current), dtype=float))

    final_index = int(np.argmin(costs[-1]))
    path = [final_index]
    for index in range(len(candidate_sets) - 1, 0, -1):
        final_index = int(backpointers[index][final_index])
        path.append(final_index)
    path.reverse()

    rr = np.array(
        [candidate_sets[i][path[i]]["rr_bpm"] for i in range(len(path))],
        dtype=float,
    )
    selected_score = np.array(
        [candidate_sets[i][path[i]]["score"] for i in range(len(path))],
        dtype=float,
    )
    emission_weight = np.array(
        [_emission_weight(float(confidence[i])) for i in range(len(path))],
        dtype=float,
    )
    selected_transition = np.array(
        [transition_costs[i][path[i]] for i in range(len(path))],
        dtype=float,
    )
    selected_harmonic = np.array(
        [transition_harmonic[i][path[i]] for i in range(len(path))],
        dtype=bool,
    )
    selected_relation = np.array(
        [transition_relation[i][path[i]] for i in range(len(path))],
        dtype=object,
    )
    selected_prior = np.array(
        [prior_costs[i][path[i]] for i in range(len(path))],
        dtype=float,
    )

    return {
        "rr_bpm": rr,
        "selected_score": selected_score,
        "emission_weight": emission_weight,
        "transition_cost": selected_transition,
        "harmonic_transition": selected_harmonic,
        "transition_relation": selected_relation,
        "initial_prior_cost": selected_prior,
        "path_index": np.asarray(path, dtype=int),
    }


def _harmonic_relation_label(raw_rr: float, tracked_rr: float) -> str:
    if not (
        np.isfinite(raw_rr)
        and np.isfinite(tracked_rr)
        and raw_rr > 0
        and tracked_rr > 0
    ):
        return ""
    if abs(raw_rr - tracked_rr) <= 0.5:
        return ""
    is_harmonic, multiple, _ = _near_integer_harmonic(raw_rr, tracked_rr)
    if not is_harmonic:
        return "nonharmonic"
    if raw_rr > tracked_rr:
        return f"{multiple}x_to_fundamental"
    return f"fundamental_to_{multiple}x"


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def apply_temporal_tracking_v4(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Apply v4 after v3, preserving every v2/v3 diagnostic series."""
    metrics_path = output_dir / "metrics.json"
    windows_path = output_dir / "windowed_rr.csv"
    if not metrics_path.exists() or not windows_path.exists():
        raise FileNotFoundError(
            "Temporal v4 requires metrics.json and windowed_rr.csv after v3."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path)
    if windows.empty:
        metrics["temporal_tracker_v4"] = {
            "version": TEMPORAL_TRACKER_V4_VERSION,
            "applied": False,
            "reason": "no sliding windows",
        }
        metrics_path.write_text(
            json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return metrics

    metadata = load_metadata(acc_path.parent)
    nominal_fs = _finite_float(metadata.get("acc_sample_rate_hz"))
    nominal_fs_arg = nominal_fs if np.isfinite(nominal_fs) else None
    raw_df = read_acc_csv(acc_path)
    segment_df, ts_qc = timestamp_qc_and_segment(raw_df, nominal_fs_arg, cfg)
    fs_hz = ts_qc.nominal_fs_hz
    _, xyz_resampled, _ = resample_uniform(segment_df, fs_hz)
    clean_xyz, _ = hampel_vector_despike(
        xyz_resampled, fs_hz, cfg.hampel_window_s, cfg.hampel_n_sigma
    )
    gravity_mg, gravity_unit = estimate_gravity(clean_xyz)
    projected = project_perpendicular_to_gravity(clean_xyz, gravity_mg, gravity_unit)
    filtered = butter_bandpass_zero_phase(
        projected, fs_hz, cfg.fmin_hz, cfg.fmax_hz, cfg.filter_order
    )

    win_n = max(16, int(round(cfg.window_s * fs_hz)))
    hop_n = max(1, int(round(cfg.hop_s * fs_hz)))
    starts = (
        list(range(0, len(filtered) - win_n + 1, hop_n))
        if len(filtered) >= win_n
        else []
    )
    if len(starts) != len(windows):
        raise RuntimeError(
            f"Temporal v4 window mismatch: recomputed {len(starts)} windows "
            f"but prior stages wrote {len(windows)}."
        )

    candidate_sets: list[list[dict[str, float]]] = []
    for row_index, start in enumerate(starts):
        stop = start + win_n
        raw_rr = float(windows.loc[row_index, "rr_final_raw_bpm"])
        try:
            local_surrogate, local_eigvals, _, local_ratio = pca_surrogate(
                filtered[start:stop]
            )
            local_move_q, _, _ = motion_quality(projected[start:stop], fs_hz, cfg)
            _, local_psd, local_acf = estimate_surrogate(
                local_surrogate,
                local_eigvals,
                local_ratio,
                fs_hz,
                cfg,
                timestamp_quality=ts_qc.timestamp_quality,
                movement_quality=local_move_q,
            )
            candidates = temporal_candidate_set(local_psd, local_acf, cfg, raw_rr)
        except Exception:
            confidence_value = float(windows.loc[row_index, "confidence"])
            fallback_score = (
                max(0.05, min(1.0, confidence_value))
                if np.isfinite(confidence_value)
                else 0.05
            )
            candidates = [
                {
                    "rr_bpm": raw_rr,
                    "score": fallback_score,
                    "spectral_support": 0.0,
                    "acf_support": 0.0,
                    "harmonic_support": 0.0,
                    "soft_alias_bonus": 0.0,
                }
            ]
        candidate_sets.append(candidates)

    confidence = windows["confidence"].to_numpy(float)
    full_estimate = metrics.get("full_record_estimate", {})
    global_prior_rr = float(full_estimate.get("rr_final_bpm", float("nan")))
    global_prior_confidence = float(full_estimate.get("confidence", float("nan")))

    track = harmonic_viterbi_track_v4(
        candidate_sets,
        confidence,
        global_prior_rr_bpm=global_prior_rr,
        global_prior_confidence=global_prior_confidence,
        hop_s=cfg.hop_s,
    )

    raw_rr = windows["rr_final_raw_bpm"].to_numpy(float)

    # Preserve v3 before the generic temporal/smoothed columns become v4.
    if "rr_temporal_bpm" in windows.columns:
        windows["rr_temporal_bpm_v3"] = windows["rr_temporal_bpm"]
    if "rr_smoothed_bpm" in windows.columns:
        windows["rr_smoothed_bpm_v3"] = windows["rr_smoothed_bpm"]
    for old, new in (
        ("temporal_selected_score", "temporal_v3_selected_score"),
        ("temporal_transition_cost", "temporal_v3_transition_cost"),
        ("temporal_harmonic_transition", "temporal_v3_harmonic_transition"),
        ("temporal_changed_from_raw", "temporal_v3_changed_from_raw"),
        ("temporal_relation", "temporal_v3_relation"),
    ):
        if old in windows.columns:
            windows[new] = windows[old]

    tracked_rr = track["rr_bpm"]
    windows["rr_temporal_bpm"] = tracked_rr
    windows["temporal_selected_score"] = track["selected_score"]
    windows["temporal_emission_weight"] = track["emission_weight"]
    windows["temporal_transition_cost"] = track["transition_cost"]
    windows["temporal_harmonic_transition"] = track["harmonic_transition"]
    windows["temporal_transition_relation"] = track["transition_relation"]
    windows["temporal_initial_prior_cost"] = track["initial_prior_cost"]
    windows["temporal_candidate_count"] = [len(candidates) for candidates in candidate_sets]
    windows["temporal_changed_from_raw"] = np.abs(tracked_rr - raw_rr) > 0.5
    windows["temporal_relation"] = [
        _harmonic_relation_label(raw, tracked)
        for raw, tracked in zip(raw_rr, tracked_rr)
    ]
    if "rr_temporal_bpm_v3" in windows.columns:
        windows["temporal_changed_from_v3"] = (
            np.abs(tracked_rr - windows["rr_temporal_bpm_v3"].to_numpy(float)) > 0.5
        )

    windows["rr_smoothed_bpm"] = kalman_track_rr(
        tracked_rr,
        confidence,
        cfg.hop_s,
        cfg,
    )

    target_rpm = float(metrics.get("target_rpm", float("nan")))
    if np.isfinite(target_rpm):
        windows["error_temporal_bpm"] = windows["rr_temporal_bpm"] - target_rpm
        windows["error_smoothed_bpm"] = windows["rr_smoothed_bpm"] - target_rpm

    windows.to_csv(windows_path, index=False)

    valid_mask = (
        windows["valid"].to_numpy(bool)
        if "valid" in windows.columns
        else np.ones(len(windows), dtype=bool)
    )
    valid = windows.loc[valid_mask]
    ws = dict(metrics.get("window_stats", {}))

    # Preserve all generic v3 statistics under explicit v3_ names.
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
        "temporal_corrections_count",
        "temporal_correction_fraction",
        "temporal_harmonic_corrections_count",
    ):
        if key in ws and f"v3_{key}" not in ws:
            ws[f"v3_{key}"] = ws[key]

    temporal_values = valid["rr_temporal_bpm"].to_numpy(float)
    smoothed_values = valid["rr_smoothed_bpm"].to_numpy(float)
    temporal_stats = _stats(temporal_values)
    smoothed_stats = _stats(smoothed_values)

    corrections = windows["temporal_changed_from_raw"].to_numpy(bool)
    relation = windows["temporal_relation"].astype(str)
    ws.update(
        {
            "rr_temporal_mean_bpm": temporal_stats["mean"],
            "rr_temporal_median_bpm": temporal_stats["median"],
            "rr_temporal_std_bpm": temporal_stats["std"],
            "rr_smoothed_mean_bpm": smoothed_stats["mean"],
            "rr_smoothed_median_bpm": smoothed_stats["median"],
            "rr_smoothed_std_bpm": smoothed_stats["std"],
            "temporal_corrections_count": int(corrections.sum()),
            "temporal_correction_fraction": float(corrections.mean()),
            "temporal_harmonic_corrections_count": int(
                relation.isin(["2x_to_fundamental", "3x_to_fundamental"]).sum()
            ),
        }
    )

    if np.isfinite(target_rpm) and len(valid):
        temporal_error = temporal_values - target_rpm
        smoothed_error = smoothed_values - target_rpm
        ws.update(
            {
                "mae_temporal_bpm": float(np.nanmean(np.abs(temporal_error))),
                "rmse_temporal_bpm": float(np.sqrt(np.nanmean(temporal_error**2))),
                "bias_temporal_bpm": float(np.nanmean(temporal_error)),
                "mae_smoothed_bpm": float(np.nanmean(np.abs(smoothed_error))),
                "rmse_smoothed_bpm": float(np.sqrt(np.nanmean(smoothed_error**2))),
                "bias_smoothed_bpm": float(np.nanmean(smoothed_error)),
            }
        )

    metrics["window_stats"] = ws
    if "temporal_tracker" in metrics and "temporal_tracker_v3" not in metrics:
        metrics["temporal_tracker_v3"] = metrics["temporal_tracker"]
    metrics["temporal_tracker"] = {
        "version": TEMPORAL_TRACKER_V4_VERSION,
        "applied": True,
        "global_prior_rr_bpm": global_prior_rr,
        "global_prior_confidence": global_prior_confidence,
        "global_prior_max_cost": V4_GLOBAL_PRIOR_MAX_COST,
        "emission_weight_floor": V4_EMISSION_WEIGHT_FLOOR,
        "continuity_penalty_per_bpm": V4_CONTINUITY_PENALTY_PER_BPM,
        "upward_harmonic_penalty": V4_UPWARD_HARMONIC_PENALTY,
        "downward_harmonic_penalty_min": V4_DOWNWARD_HARMONIC_PENALTY_MIN,
        "downward_harmonic_penalty_max": V4_DOWNWARD_HARMONIC_PENALTY_MAX,
        "corrections_count": int(corrections.sum()),
        "harmonic_corrections_count": int(
            relation.isin(["2x_to_fundamental", "3x_to_fundamental"]).sum()
        ),
        "changed_from_v3_count": int(
            windows.get(
                "temporal_changed_from_v3",
                pd.Series(False, index=windows.index),
            ).sum()
        ),
    }
    metrics["temporal_tracker_v4"] = metrics["temporal_tracker"]

    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _plot_temporal_v4(
        output_dir / "temporal_v4.png",
        windows,
        target_rpm,
        global_prior_rr,
        cfg,
    )
    return metrics


def _plot_temporal_v4(
    path: Path,
    windows: pd.DataFrame,
    target_rpm: float,
    global_prior_rr: float,
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    time_s = windows["time_s"]

    axes[0].plot(time_s, windows["rr_final_raw_bpm"], label="Local v2", alpha=0.6)
    if "rr_temporal_bpm_v3" in windows.columns:
        axes[0].plot(
            time_s,
            windows["rr_temporal_bpm_v3"],
            label="Temporal v3",
            alpha=0.65,
        )
    axes[0].plot(time_s, windows["rr_temporal_bpm"], label="Temporal v4")
    if "rr_smoothed_bpm_v3" in windows.columns:
        axes[0].plot(
            time_s,
            windows["rr_smoothed_bpm_v3"],
            label="v3 + Kalman",
            alpha=0.6,
        )
    axes[0].plot(time_s, windows["rr_smoothed_bpm"], label="v4 + Kalman")
    if np.isfinite(global_prior_rr):
        axes[0].axhline(
            global_prior_rr,
            linestyle=":",
            label=f"Prior global {global_prior_rr:.2f}",
        )
    if np.isfinite(target_rpm):
        axes[0].axhline(target_rpm, linestyle="--", label=f"Alvo {target_rpm:g}")
    axes[0].set_ylabel("RR [resp/min]")
    axes[0].set_title(
        f"Temporal v4 — janelas {cfg.window_s:g}s / hop {cfg.hop_s:g}s"
    )
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(time_s, windows["confidence"], label="Confiança local")
    axes[1].plot(
        time_s,
        windows["temporal_emission_weight"],
        label="Peso da emissão v4",
    )
    axes[1].plot(
        time_s,
        windows["temporal_selected_score"],
        label="Score selecionado v4",
    )
    axes[1].set_xlabel("Tempo [s]")
    axes[1].set_ylabel("Score / peso")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Confiança, emissão e estado temporal v4")

    fig.savefig(path, dpi=160)
    plt.close(fig)
