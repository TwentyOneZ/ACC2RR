from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    Config,
    EPS,
    _finite_float,
    _jsonable,
    butter_bandpass_zero_phase,
    estimate_gravity,
    hampel_vector_despike,
    load_metadata,
    project_perpendicular_to_gravity,
    read_acc_csv,
    resample_uniform,
    timestamp_qc_and_segment,
)
from .estimators import kalman_track_rr
from .surrogate_v5 import PC12_ANGLE_STEP_DEG, evaluate_surrogate
from .temporal import HARMONIC_RATIO_TOLERANCE

TEMPORAL_TRACKER_V6_VERSION = "joint_rr_orientation_v1"
V6_RR_CONTINUITY_PENALTY_PER_BPM = 0.018
V6_UPWARD_HARMONIC_PENALTY = 6.0
V6_DOWNWARD_HARMONIC_PENALTY_MIN = 1.2
V6_DOWNWARD_HARMONIC_PENALTY_MAX = 3.0
V6_GLOBAL_PRIOR_MAX_COST = 4.5
V6_EMISSION_WEIGHT_FLOOR = 0.25
V6_ANGULAR_PENALTY_PER_RAD = 0.30


def _near_integer_harmonic(rr_a: float, rr_b: float) -> tuple[bool, int, float]:
    if not (np.isfinite(rr_a) and np.isfinite(rr_b) and rr_a > 0 and rr_b > 0):
        return False, 1, float("inf")
    ratio = max(rr_a, rr_b) / min(rr_a, rr_b)
    multiple = min((2, 3), key=lambda m: abs(ratio - m))
    rel_error = abs(ratio - multiple) / multiple
    return rel_error <= HARMONIC_RATIO_TOLERANCE, int(multiple), float(rel_error)


def orientation_distance_rad(direction_a: np.ndarray, direction_b: np.ndarray) -> float:
    """Angle between unoriented 3-D axes, treating v and -v as equivalent."""
    a = np.asarray(direction_a, dtype=float)
    b = np.asarray(direction_b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < EPS or norm_b < EPS:
        return 0.0
    cosine = float(np.clip(abs(np.dot(a / norm_a, b / norm_b)), 0.0, 1.0))
    return float(math.acos(cosine))


def _emission_weight(confidence: float) -> float:
    value = float(np.clip(confidence, 0.0, 1.0)) if np.isfinite(confidence) else 0.0
    return V6_EMISSION_WEIGHT_FLOOR + (1.0 - V6_EMISSION_WEIGHT_FLOOR) * value


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
        V6_GLOBAL_PRIOR_MAX_COST
        * confidence
        * min(normalized * normalized, 1.0)
    )


def _window_pca_basis(
    filtered_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centered = filtered_window - np.mean(filtered_window, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    components = centered @ eigvecs
    total = float(np.sum(eigvals))
    ratios = eigvals / total if total > EPS else np.zeros_like(eigvals)
    return components, eigvals, eigvecs, ratios


def build_v6_states_for_window(
    filtered_window: np.ndarray,
    fs_hz: float,
    cfg: Config,
    reference_rr_bpm: float,
) -> list[dict[str, float]]:
    """Build one RR+physical-direction state for every PC1-PC2 scan angle."""
    components, _, eigvecs, ratios = _window_pca_basis(filtered_window)
    pc1 = components[:, 0]
    pc2 = components[:, 1]

    states: list[dict[str, float]] = []
    for angle_deg in range(0, 180, PC12_ANGLE_STEP_DEG):
        theta = math.radians(angle_deg)
        cosine = math.cos(theta)
        sine = math.sin(theta)
        surrogate = pc1 * cosine + pc2 * sine
        evaluation = evaluate_surrogate(
            surrogate,
            fs_hz,
            cfg,
            reference_rr_bpm=reference_rr_bpm,
        )
        rr_bpm = float(evaluation.get("rr_consensus_bpm", float("nan")))
        confidence = float(evaluation.get("benchmark_confidence", 0.0))
        if not (np.isfinite(rr_bpm) and rr_bpm > 0 and np.isfinite(confidence)):
            continue

        direction = eigvecs[:, 0] * cosine + eigvecs[:, 1] * sine
        norm = float(np.linalg.norm(direction))
        if norm < EPS:
            continue
        direction = direction / norm

        states.append(
            {
                "angle_deg": float(angle_deg),
                "rr_bpm": rr_bpm,
                "score": confidence,
                "benchmark_confidence": confidence,
                "spectral_quality": float(evaluation.get("spectral_quality", 0.0)),
                "acf_quality": float(evaluation.get("acf_quality", 0.0)),
                "agreement_quality": float(evaluation.get("agreement_quality", 0.0)),
                "consensus_score": float(evaluation.get("consensus_score", 0.0)),
                "reference_fundamental_to_2x_ratio": float(
                    evaluation.get("reference_fundamental_to_2x_ratio", float("nan"))
                ),
                "pca_pc1_variance_ratio": float(ratios[0]),
                "pca_pc2_variance_ratio": float(ratios[1]),
                "direction_x": float(direction[0]),
                "direction_y": float(direction[1]),
                "direction_z": float(direction[2]),
            }
        )
    return states


def _direction_from_state(state: dict[str, float]) -> np.ndarray:
    return np.array(
        [state["direction_x"], state["direction_y"], state["direction_z"]],
        dtype=float,
    )


def _transition_cost_v6(
    previous_state: dict[str, float],
    current_state: dict[str, float],
    hop_s: float,
) -> tuple[float, float, float, float, float, bool, str]:
    hop_scale = max(float(hop_s), 1.0)
    previous_rr = float(previous_state["rr_bpm"])
    current_rr = float(current_state["rr_bpm"])

    rr_cost = (
        V6_RR_CONTINUITY_PENALTY_PER_BPM
        * abs(current_rr - previous_rr)
        / hop_scale
    )

    harmonic_cost = 0.0
    harmonic_jump, multiple, _ = _near_integer_harmonic(previous_rr, current_rr)
    relation = ""
    if harmonic_jump:
        acf_quality = float(np.clip(current_state.get("acf_quality", 0.0), 0.0, 1.0))
        confidence = float(
            np.clip(current_state.get("benchmark_confidence", 0.0), 0.0, 1.0)
        )
        if current_rr > previous_rr:
            evidence = float(np.clip((acf_quality * confidence) / 0.55, 0.0, 1.0))
            harmonic_penalty = V6_UPWARD_HARMONIC_PENALTY * (1.0 - 0.55 * evidence)
            relation = f"fundamental_to_{multiple}x"
        else:
            harmonic_penalty = (
                V6_DOWNWARD_HARMONIC_PENALTY_MIN
                + (V6_DOWNWARD_HARMONIC_PENALTY_MAX - V6_DOWNWARD_HARMONIC_PENALTY_MIN)
                * (1.0 - acf_quality)
            )
            relation = f"{multiple}x_to_fundamental"
        harmonic_cost = harmonic_penalty / hop_scale

    angular_distance = orientation_distance_rad(
        _direction_from_state(previous_state),
        _direction_from_state(current_state),
    )
    angular_cost = V6_ANGULAR_PENALTY_PER_RAD * angular_distance / hop_scale
    total = rr_cost + harmonic_cost + angular_cost
    return (
        float(total),
        float(rr_cost),
        float(harmonic_cost),
        float(angular_cost),
        float(angular_distance),
        bool(harmonic_jump),
        relation,
    )


def joint_viterbi_track_v6(
    state_sets: list[list[dict[str, float]]],
    global_prior_rr_bpm: float,
    global_prior_confidence: float,
    hop_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Track RR and respiratory orientation jointly across overlapping windows."""
    if not state_sets:
        empty = np.array([], dtype=float)
        return {
            "rr_bpm": empty,
            "angle_deg": empty,
            "benchmark_confidence": empty,
            "transition_cost": empty,
            "rr_transition_cost": empty,
            "harmonic_transition_cost": empty,
            "angular_transition_cost": empty,
            "angular_distance_rad": empty,
            "harmonic_transition": np.array([], dtype=bool),
            "transition_relation": np.array([], dtype=object),
            "initial_prior_cost": empty,
            "path_index": np.array([], dtype=int),
        }
    if any(len(states) == 0 for states in state_sets):
        raise ValueError("Every v6 temporal window must contain at least one state.")

    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    total_transition_costs: list[np.ndarray] = []
    rr_transition_costs: list[np.ndarray] = []
    harmonic_transition_costs: list[np.ndarray] = []
    angular_transition_costs: list[np.ndarray] = []
    angular_distances: list[np.ndarray] = []
    harmonic_flags: list[np.ndarray] = []
    transition_relations: list[np.ndarray] = []
    prior_costs: list[np.ndarray] = []

    first = state_sets[0]
    first_prior = np.array(
        [
            _initial_prior_cost(
                float(state["rr_bpm"]),
                global_prior_rr_bpm,
                global_prior_confidence,
            )
            for state in first
        ],
        dtype=float,
    )
    first_cost = np.array(
        [
            -float(state["score"]) * _emission_weight(float(state["score"]))
            + first_prior[index]
            for index, state in enumerate(first)
        ],
        dtype=float,
    )
    costs.append(first_cost)
    backpointers.append(np.full(len(first), -1, dtype=int))
    total_transition_costs.append(np.zeros(len(first), dtype=float))
    rr_transition_costs.append(np.zeros(len(first), dtype=float))
    harmonic_transition_costs.append(np.zeros(len(first), dtype=float))
    angular_transition_costs.append(np.zeros(len(first), dtype=float))
    angular_distances.append(np.zeros(len(first), dtype=float))
    harmonic_flags.append(np.zeros(len(first), dtype=bool))
    transition_relations.append(np.full(len(first), "", dtype=object))
    prior_costs.append(first_prior)

    for window_index in range(1, len(state_sets)):
        previous = state_sets[window_index - 1]
        current = state_sets[window_index]
        current_cost = np.full(len(current), np.inf, dtype=float)
        current_back = np.full(len(current), -1, dtype=int)
        current_total_transition = np.full(len(current), np.nan, dtype=float)
        current_rr_transition = np.full(len(current), np.nan, dtype=float)
        current_harmonic_transition_cost = np.full(len(current), np.nan, dtype=float)
        current_angular_transition = np.full(len(current), np.nan, dtype=float)
        current_angular_distance = np.full(len(current), np.nan, dtype=float)
        current_harmonic_flag = np.zeros(len(current), dtype=bool)
        current_relation = np.full(len(current), "", dtype=object)

        for current_index, state in enumerate(current):
            score = float(state["score"])
            emission_cost = -score * _emission_weight(score)
            for previous_index, previous_state in enumerate(previous):
                (
                    transition_cost,
                    rr_cost,
                    harmonic_cost,
                    angular_cost,
                    angular_distance,
                    harmonic_flag,
                    relation,
                ) = _transition_cost_v6(previous_state, state, hop_s)
                total = costs[window_index - 1][previous_index] + transition_cost + emission_cost
                if total < current_cost[current_index]:
                    current_cost[current_index] = total
                    current_back[current_index] = previous_index
                    current_total_transition[current_index] = transition_cost
                    current_rr_transition[current_index] = rr_cost
                    current_harmonic_transition_cost[current_index] = harmonic_cost
                    current_angular_transition[current_index] = angular_cost
                    current_angular_distance[current_index] = angular_distance
                    current_harmonic_flag[current_index] = harmonic_flag
                    current_relation[current_index] = relation

        costs.append(current_cost)
        backpointers.append(current_back)
        total_transition_costs.append(current_total_transition)
        rr_transition_costs.append(current_rr_transition)
        harmonic_transition_costs.append(current_harmonic_transition_cost)
        angular_transition_costs.append(current_angular_transition)
        angular_distances.append(current_angular_distance)
        harmonic_flags.append(current_harmonic_flag)
        transition_relations.append(current_relation)
        prior_costs.append(np.zeros(len(current), dtype=float))

    final_index = int(np.argmin(costs[-1]))
    path = [final_index]
    for window_index in range(len(state_sets) - 1, 0, -1):
        final_index = int(backpointers[window_index][final_index])
        path.append(final_index)
    path.reverse()

    selected = [state_sets[i][path[i]] for i in range(len(path))]
    return {
        "rr_bpm": np.array([state["rr_bpm"] for state in selected], dtype=float),
        "angle_deg": np.array([state["angle_deg"] for state in selected], dtype=float),
        "benchmark_confidence": np.array(
            [state["benchmark_confidence"] for state in selected], dtype=float
        ),
        "spectral_quality": np.array(
            [state["spectral_quality"] for state in selected], dtype=float
        ),
        "acf_quality": np.array([state["acf_quality"] for state in selected], dtype=float),
        "agreement_quality": np.array(
            [state["agreement_quality"] for state in selected], dtype=float
        ),
        "consensus_score": np.array(
            [state["consensus_score"] for state in selected], dtype=float
        ),
        "reference_fundamental_to_2x_ratio": np.array(
            [state["reference_fundamental_to_2x_ratio"] for state in selected], dtype=float
        ),
        "direction_x": np.array([state["direction_x"] for state in selected], dtype=float),
        "direction_y": np.array([state["direction_y"] for state in selected], dtype=float),
        "direction_z": np.array([state["direction_z"] for state in selected], dtype=float),
        "transition_cost": np.array(
            [total_transition_costs[i][path[i]] for i in range(len(path))], dtype=float
        ),
        "rr_transition_cost": np.array(
            [rr_transition_costs[i][path[i]] for i in range(len(path))], dtype=float
        ),
        "harmonic_transition_cost": np.array(
            [harmonic_transition_costs[i][path[i]] for i in range(len(path))], dtype=float
        ),
        "angular_transition_cost": np.array(
            [angular_transition_costs[i][path[i]] for i in range(len(path))], dtype=float
        ),
        "angular_distance_rad": np.array(
            [angular_distances[i][path[i]] for i in range(len(path))], dtype=float
        ),
        "harmonic_transition": np.array(
            [harmonic_flags[i][path[i]] for i in range(len(path))], dtype=bool
        ),
        "transition_relation": np.array(
            [transition_relations[i][path[i]] for i in range(len(path))], dtype=object
        ),
        "initial_prior_cost": np.array(
            [prior_costs[i][path[i]] for i in range(len(path))], dtype=float
        ),
        "path_index": np.asarray(path, dtype=int),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def _error_stats(values: np.ndarray, target_rpm: float) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not (len(finite) and np.isfinite(target_rpm)):
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
        }
    error = finite - target_rpm
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def apply_joint_tracking_v6(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Integrate the v5 PC1-PC2 scan with v4-style temporal tracking."""
    metrics_path = output_dir / "metrics.json"
    windows_path = output_dir / "windowed_rr.csv"
    if not metrics_path.exists() or not windows_path.exists():
        raise FileNotFoundError(
            "Temporal v6 requires metrics.json and windowed_rr.csv after v4/v5."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path)
    if windows.empty:
        metrics["temporal_tracker_v6"] = {
            "version": TEMPORAL_TRACKER_V6_VERSION,
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
            f"Temporal v6 window mismatch: recomputed {len(starts)} windows "
            f"but prior stages wrote {len(windows)}."
        )

    full_estimate = metrics.get("full_record_estimate", {})
    global_prior_rr = float(full_estimate.get("rr_final_bpm", float("nan")))
    global_prior_confidence = float(full_estimate.get("confidence", float("nan")))

    state_sets: list[list[dict[str, float]]] = []
    state_rows: list[dict[str, Any]] = []
    for window_index, start in enumerate(starts):
        stop = start + win_n
        states = build_v6_states_for_window(
            filtered[start:stop],
            fs_hz,
            cfg,
            reference_rr_bpm=global_prior_rr,
        )
        if not states:
            fallback_rr = float(windows.loc[window_index, "rr_temporal_bpm"])
            fallback_confidence = float(windows.loc[window_index, "confidence"])
            components, _, eigvecs, ratios = _window_pca_basis(filtered[start:stop])
            del components
            direction = eigvecs[:, 0]
            states = [
                {
                    "angle_deg": 0.0,
                    "rr_bpm": fallback_rr,
                    "score": max(0.05, min(1.0, fallback_confidence)),
                    "benchmark_confidence": max(0.05, min(1.0, fallback_confidence)),
                    "spectral_quality": 0.0,
                    "acf_quality": 0.0,
                    "agreement_quality": 0.0,
                    "consensus_score": 0.0,
                    "reference_fundamental_to_2x_ratio": float("nan"),
                    "pca_pc1_variance_ratio": float(ratios[0]),
                    "pca_pc2_variance_ratio": float(ratios[1]),
                    "direction_x": float(direction[0]),
                    "direction_y": float(direction[1]),
                    "direction_z": float(direction[2]),
                }
            ]
        state_sets.append(states)
        for state_index, state in enumerate(states):
            row = {
                "window_index": int(window_index),
                "state_index": int(state_index),
                "time_s": float(windows.loc[window_index, "time_s"]),
                "timestamp_epoch_s": float(windows.loc[window_index, "timestamp_epoch_s"]),
            }
            row.update(state)
            state_rows.append(row)

    track = joint_viterbi_track_v6(
        state_sets,
        global_prior_rr_bpm=global_prior_rr,
        global_prior_confidence=global_prior_confidence,
        hop_s=cfg.hop_s,
    )

    # Preserve v4 before the generic temporal/smoothed columns become v6.
    if "rr_temporal_bpm" in windows.columns:
        windows["rr_temporal_bpm_v4"] = windows["rr_temporal_bpm"]
    if "rr_smoothed_bpm" in windows.columns:
        windows["rr_smoothed_bpm_v4"] = windows["rr_smoothed_bpm"]
    for old, new in (
        ("temporal_selected_score", "temporal_v4_selected_score"),
        ("temporal_emission_weight", "temporal_v4_emission_weight"),
        ("temporal_transition_cost", "temporal_v4_transition_cost"),
        ("temporal_harmonic_transition", "temporal_v4_harmonic_transition"),
        ("temporal_transition_relation", "temporal_v4_transition_relation"),
        ("temporal_initial_prior_cost", "temporal_v4_initial_prior_cost"),
        ("temporal_changed_from_raw", "temporal_v4_changed_from_raw"),
        ("temporal_relation", "temporal_v4_relation"),
    ):
        if old in windows.columns:
            windows[new] = windows[old]

    tracked_rr = track["rr_bpm"]
    selected_confidence = track["benchmark_confidence"]
    windows["rr_temporal_bpm"] = tracked_rr
    windows["v6_selected_angle_deg"] = track["angle_deg"]
    windows["v6_direction_x"] = track["direction_x"]
    windows["v6_direction_y"] = track["direction_y"]
    windows["v6_direction_z"] = track["direction_z"]
    windows["v6_benchmark_confidence"] = selected_confidence
    windows["v6_spectral_quality"] = track["spectral_quality"]
    windows["v6_acf_quality"] = track["acf_quality"]
    windows["v6_agreement_quality"] = track["agreement_quality"]
    windows["v6_consensus_score"] = track["consensus_score"]
    windows["v6_reference_fundamental_to_2x_ratio"] = track[
        "reference_fundamental_to_2x_ratio"
    ]
    windows["v6_transition_cost"] = track["transition_cost"]
    windows["v6_rr_transition_cost"] = track["rr_transition_cost"]
    windows["v6_harmonic_transition_cost"] = track["harmonic_transition_cost"]
    windows["v6_angular_transition_cost"] = track["angular_transition_cost"]
    windows["v6_angular_step_deg"] = np.degrees(track["angular_distance_rad"])
    windows["v6_harmonic_transition"] = track["harmonic_transition"]
    windows["v6_transition_relation"] = track["transition_relation"]
    windows["v6_initial_prior_cost"] = track["initial_prior_cost"]
    windows["v6_state_count"] = [len(states) for states in state_sets]
    windows["v6_valid"] = selected_confidence >= cfg.min_confidence

    raw_rr = windows["rr_final_raw_bpm"].to_numpy(float)
    windows["v6_changed_from_raw"] = np.abs(tracked_rr - raw_rr) > 0.5
    if "rr_temporal_bpm_v4" in windows.columns:
        windows["v6_changed_from_v4"] = (
            np.abs(tracked_rr - windows["rr_temporal_bpm_v4"].to_numpy(float)) > 0.5
        )

    windows["rr_smoothed_bpm"] = kalman_track_rr(
        tracked_rr,
        selected_confidence,
        cfg.hop_s,
        cfg,
    )

    target_rpm = float(metrics.get("target_rpm", float("nan")))
    if np.isfinite(target_rpm):
        windows["error_temporal_bpm"] = windows["rr_temporal_bpm"] - target_rpm
        windows["error_smoothed_bpm"] = windows["rr_smoothed_bpm"] - target_rpm

    windows.to_csv(windows_path, index=False)

    selected_pairs = {
        (int(window_index), int(state_index))
        for window_index, state_index in enumerate(track["path_index"])
    }
    states_df = pd.DataFrame(state_rows)
    states_df["selected"] = [
        (int(row.window_index), int(row.state_index)) in selected_pairs
        for row in states_df.itertuples(index=False)
    ]
    states_df.to_csv(output_dir / "temporal_v6_states.csv", index=False)

    ws = dict(metrics.get("window_stats", {}))
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
        if key in ws and f"v4_{key}" not in ws:
            ws[f"v4_{key}"] = ws[key]

    legacy_valid_mask = (
        windows["valid"].to_numpy(bool)
        if "valid" in windows.columns
        else np.ones(len(windows), dtype=bool)
    )
    v6_valid_mask = windows["v6_valid"].to_numpy(bool)
    temporal_legacy = tracked_rr[legacy_valid_mask]
    smooth_all = windows["rr_smoothed_bpm"].to_numpy(float)
    smooth_legacy = smooth_all[legacy_valid_mask]

    temporal_stats = _stats(temporal_legacy)
    smooth_stats = _stats(smooth_legacy)
    ws.update(
        {
            "rr_temporal_mean_bpm": temporal_stats["mean"],
            "rr_temporal_median_bpm": temporal_stats["median"],
            "rr_temporal_std_bpm": temporal_stats["std"],
            "rr_smoothed_mean_bpm": smooth_stats["mean"],
            "rr_smoothed_median_bpm": smooth_stats["median"],
            "rr_smoothed_std_bpm": smooth_stats["std"],
            "v6_selected_confidence_mean": float(np.nanmean(selected_confidence)),
            "v6_selected_confidence_median": float(np.nanmedian(selected_confidence)),
            "v6_selected_angle_median_deg": float(np.nanmedian(track["angle_deg"])),
            "v6_angular_step_mean_deg": float(np.nanmean(np.degrees(track["angular_distance_rad"]))),
            "v6_angular_step_median_deg": float(np.nanmedian(np.degrees(track["angular_distance_rad"]))),
            "v6_selected_valid_fraction": float(np.mean(v6_valid_mask)),
            "v6_changed_from_raw_count": int(windows["v6_changed_from_raw"].sum()),
            "v6_changed_from_v4_count": int(
                windows.get("v6_changed_from_v4", pd.Series(False, index=windows.index)).sum()
            ),
        }
    )

    if np.isfinite(target_rpm):
        legacy_temporal_error = _error_stats(temporal_legacy, target_rpm)
        legacy_smooth_error = _error_stats(smooth_legacy, target_rpm)
        all_temporal_error = _error_stats(tracked_rr, target_rpm)
        all_smooth_error = _error_stats(smooth_all, target_rpm)
        selected_valid_temporal_error = _error_stats(tracked_rr[v6_valid_mask], target_rpm)
        selected_valid_smooth_error = _error_stats(smooth_all[v6_valid_mask], target_rpm)
        ws.update(
            {
                "mae_temporal_bpm": legacy_temporal_error["mae"],
                "rmse_temporal_bpm": legacy_temporal_error["rmse"],
                "bias_temporal_bpm": legacy_temporal_error["bias"],
                "mae_smoothed_bpm": legacy_smooth_error["mae"],
                "rmse_smoothed_bpm": legacy_smooth_error["rmse"],
                "bias_smoothed_bpm": legacy_smooth_error["bias"],
                "v6_all_mae_temporal_bpm": all_temporal_error["mae"],
                "v6_all_rmse_temporal_bpm": all_temporal_error["rmse"],
                "v6_all_bias_temporal_bpm": all_temporal_error["bias"],
                "v6_all_mae_smoothed_bpm": all_smooth_error["mae"],
                "v6_all_rmse_smoothed_bpm": all_smooth_error["rmse"],
                "v6_all_bias_smoothed_bpm": all_smooth_error["bias"],
                "v6_selected_valid_mae_temporal_bpm": selected_valid_temporal_error["mae"],
                "v6_selected_valid_rmse_temporal_bpm": selected_valid_temporal_error["rmse"],
                "v6_selected_valid_bias_temporal_bpm": selected_valid_temporal_error["bias"],
                "v6_selected_valid_mae_smoothed_bpm": selected_valid_smooth_error["mae"],
                "v6_selected_valid_rmse_smoothed_bpm": selected_valid_smooth_error["rmse"],
                "v6_selected_valid_bias_smoothed_bpm": selected_valid_smooth_error["bias"],
            }
        )

    metrics["window_stats"] = ws
    metrics["temporal_tracker_v6"] = {
        "version": TEMPORAL_TRACKER_V6_VERSION,
        "applied": True,
        "selection_uses_target_rpm": False,
        "state_definition": "PC1*cos(theta)+PC2*sin(theta), theta=0..175 deg",
        "physical_direction_continuity": "acos(abs(v_t dot v_t-1))",
        "angle_step_deg": PC12_ANGLE_STEP_DEG,
        "states_per_window_expected": len(range(0, 180, PC12_ANGLE_STEP_DEG)),
        "global_prior_source": "full_record_estimate.rr_final_bpm",
        "global_prior_rr_bpm": global_prior_rr,
        "global_prior_confidence": global_prior_confidence,
        "global_prior_max_cost": V6_GLOBAL_PRIOR_MAX_COST,
        "rr_continuity_penalty_per_bpm": V6_RR_CONTINUITY_PENALTY_PER_BPM,
        "angular_penalty_per_rad": V6_ANGULAR_PENALTY_PER_RAD,
        "upward_harmonic_penalty": V6_UPWARD_HARMONIC_PENALTY,
        "downward_harmonic_penalty_min": V6_DOWNWARD_HARMONIC_PENALTY_MIN,
        "downward_harmonic_penalty_max": V6_DOWNWARD_HARMONIC_PENALTY_MAX,
        "emission_weight_floor": V6_EMISSION_WEIGHT_FLOOR,
        "selected_confidence_mean": float(np.nanmean(selected_confidence)),
        "selected_valid_fraction": float(np.mean(v6_valid_mask)),
        "mean_physical_angular_step_deg": float(
            np.nanmean(np.degrees(track["angular_distance_rad"]))
        ),
        "median_physical_angular_step_deg": float(
            np.nanmedian(np.degrees(track["angular_distance_rad"]))
        ),
        "changed_from_v4_count": int(
            windows.get("v6_changed_from_v4", pd.Series(False, index=windows.index)).sum()
        ),
    }
    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_temporal_v6(
        output_dir / "temporal_v6.png",
        windows,
        target_rpm,
        global_prior_rr,
        cfg,
    )
    return metrics


def _plot_temporal_v6(
    path: Path,
    windows: pd.DataFrame,
    target_rpm: float,
    global_prior_rr: float,
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    time_s = windows["time_s"]

    axes[0].plot(time_s, windows["rr_final_raw_bpm"], label="Local v2", alpha=0.35)
    if "rr_temporal_bpm_v4" in windows.columns:
        axes[0].plot(time_s, windows["rr_temporal_bpm_v4"], label="Temporal v4", alpha=0.60)
    axes[0].plot(time_s, windows["rr_temporal_bpm"], label="Joint v6")
    if "rr_smoothed_bpm_v4" in windows.columns:
        axes[0].plot(time_s, windows["rr_smoothed_bpm_v4"], label="v4 + Kalman", alpha=0.55)
    axes[0].plot(time_s, windows["rr_smoothed_bpm"], label="v6 + Kalman")
    if np.isfinite(global_prior_rr):
        axes[0].axhline(global_prior_rr, linestyle=":", label=f"Prior global {global_prior_rr:.2f}")
    if np.isfinite(target_rpm):
        axes[0].axhline(target_rpm, linestyle="--", label=f"Alvo {target_rpm:g}")
    axes[0].set_ylabel("RR [resp/min]")
    axes[0].set_title(f"Temporal v6 — RR + orientação, janela {cfg.window_s:g}s / hop {cfg.hop_s:g}s")
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(time_s, windows["v6_selected_angle_deg"], label="theta local selecionado")
    axes[1].plot(time_s, windows["v6_angular_step_deg"], label="mudança física entre janelas")
    axes[1].set_ylabel("Ângulo [graus]")
    axes[1].set_title("Orientação respiratória selecionada e continuidade física")
    axes[1].legend(loc="upper right")

    axes[2].plot(time_s, windows["v6_benchmark_confidence"], label="benchmark confidence v6")
    axes[2].plot(time_s, windows["v6_acf_quality"], label="ACF quality")
    axes[2].plot(
        time_s,
        windows["v6_reference_fundamental_to_2x_ratio"],
        label="P(f_ref)/P(2f_ref)",
        alpha=0.75,
    )
    axes[2].axhline(cfg.min_confidence, linestyle="--", label="min confidence")
    axes[2].set_xlabel("Tempo [s]")
    axes[2].set_ylabel("Score / razão")
    axes[2].set_ylim(bottom=0)
    axes[2].set_title("Evidência respiratória do estado selecionado")
    axes[2].legend(loc="upper right", ncols=2)

    fig.savefig(path, dpi=160)
    plt.close(fig)
