from __future__ import annotations

import json
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
    pca_surrogate,
    project_perpendicular_to_gravity,
    read_acc_csv,
    resample_uniform,
    timestamp_qc_and_segment,
)
from .estimators import estimate_surrogate, kalman_track_rr, motion_quality

TEMPORAL_TRACKER_VERSION = "harmonic_viterbi_v1"
CONTINUITY_PENALTY_PER_BPM = 0.018
HARMONIC_JUMP_PENALTY = 5.0
HARMONIC_RATIO_TOLERANCE = 0.08
SOFT_ALIAS_BONUS = 0.24
MAX_CANDIDATES_PER_WINDOW = 12


def _normalized_psd_at(f_hz: float, psd: dict[str, Any]) -> float:
    if not np.isfinite(f_hz) or f_hz <= 0:
        return 0.0
    p = float(np.interp(f_hz, psd["freq_hz"], psd["power"], left=0.0, right=0.0))
    return float(np.clip(p / max(float(psd.get("p_max", 0.0)), EPS), 0.0, 1.0))


def _acf_at_frequency(f_hz: float, acf: dict[str, Any]) -> float:
    if not np.isfinite(f_hz) or f_hz <= 0:
        return 0.0
    lag = 1.0 / f_hz
    value = float(np.interp(lag, acf["lags_s"], acf["acf"], left=0.0, right=0.0))
    return float(np.clip(value, 0.0, 1.0))


def _near_integer_harmonic(rr_a: float, rr_b: float) -> tuple[bool, int, float]:
    if not (np.isfinite(rr_a) and np.isfinite(rr_b) and rr_a > 0 and rr_b > 0):
        return False, 1, float("inf")
    ratio = max(rr_a, rr_b) / min(rr_a, rr_b)
    multiple = min((2, 3), key=lambda m: abs(ratio - m))
    rel_error = abs(ratio - multiple) / multiple
    return rel_error <= HARMONIC_RATIO_TOLERANCE, int(multiple), float(rel_error)


def temporal_candidate_set(
    psd: dict[str, Any],
    acf: dict[str, Any],
    cfg: Config,
    raw_rr_bpm: float,
) -> list[dict[str, float]]:
    """Build candidate RR states for one window.

    The v2 local decision is preserved as one candidate, but the tracker also
    sees nearby PSD/ACF peaks and their integer-harmonic relatives. A *soft*
    alias bonus is allowed below the strict v2 harmonic gate; it is intentionally
    insufficient on its own and only becomes useful when temporal continuity
    supports the same fundamental over neighboring windows.
    """
    source_freqs: list[float] = []
    source_freqs.extend(float(f) for f in psd.get("candidate_freqs_hz", []) if np.isfinite(f))
    source_freqs.extend(float(f) for f in acf.get("candidate_freqs_hz", []) if np.isfinite(f))

    for f in (
        psd.get("f_selected_hz", float("nan")),
        float(acf.get("rr_bpm", float("nan"))) / 60.0,
        float(raw_rr_bpm) / 60.0,
    ):
        if np.isfinite(f) and f > 0:
            source_freqs.append(float(f))

    expanded: list[float] = []
    for f in source_freqs:
        for multiplier in (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0):
            candidate = f * multiplier
            if cfg.fmin_hz <= candidate <= cfg.fmax_hz:
                expanded.append(float(candidate))

    if not expanded and np.isfinite(raw_rr_bpm) and raw_rr_bpm > 0:
        expanded = [raw_rr_bpm / 60.0]

    expanded.sort()
    merged: list[float] = []
    for f in expanded:
        if not merged or abs(f - merged[-1]) > 0.005:
            merged.append(f)
        else:
            merged[-1] = 0.5 * (merged[-1] + f)

    rr_psd = float(psd.get("rr_bpm", float("nan")))
    rr_acf = float(acf.get("rr_bpm", float("nan")))
    alias_like, alias_multiple, _ = _near_integer_harmonic(rr_psd, rr_acf)
    acf_quality = float(acf.get("quality", 0.0))
    acf_f = rr_acf / 60.0 if np.isfinite(rr_acf) and rr_acf > 0 else float("nan")

    candidates: list[dict[str, float]] = []
    for f in merged:
        spectral = _normalized_psd_at(f, psd)
        acf_support = _acf_at_frequency(f, acf)
        second = _normalized_psd_at(2.0 * f, psd) if 2.0 * f <= cfg.fmax_hz else 0.0
        third = _normalized_psd_at(3.0 * f, psd) if 3.0 * f <= cfg.fmax_hz else 0.0
        harmonic_support = max(second, 0.75 * third)

        score = 0.48 * spectral + 0.42 * acf_support + 0.10 * harmonic_support
        if spectral < 0.04 and acf_support < 0.25:
            score *= 0.5

        # Keep the already validated v2 local decision competitive.
        if np.isfinite(raw_rr_bpm) and abs(60.0 * f - raw_rr_bpm) <= 0.75:
            score += 0.06

        soft_bonus = 0.0
        if alias_like and np.isfinite(acf_f):
            tol_hz = max(0.008, 0.04 * acf_f)
            if abs(f - acf_f) <= tol_hz:
                direct = spectral
                harmonic_at_multiple = _normalized_psd_at(alias_multiple * f, psd)
                if (
                    acf_quality >= 0.35
                    and acf_support >= 0.30
                    and direct >= 0.035
                    and harmonic_at_multiple >= 0.45
                ):
                    strength = (
                        min(1.0, acf_quality / 0.50)
                        * min(1.0, acf_support / 0.40)
                        * min(1.0, direct / 0.10)
                        * min(1.0, harmonic_at_multiple / 0.50)
                    )
                    soft_bonus = SOFT_ALIAS_BONUS * strength
                    score += soft_bonus

        candidates.append(
            {
                "rr_bpm": 60.0 * f,
                "score": float(np.clip(score, 0.0, 1.5)),
                "spectral_support": spectral,
                "acf_support": acf_support,
                "harmonic_support": harmonic_support,
                "soft_alias_bonus": soft_bonus,
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    kept = candidates[:MAX_CANDIDATES_PER_WINDOW]

    # Never drop the v2 result simply because many spectral side peaks exist.
    if np.isfinite(raw_rr_bpm) and raw_rr_bpm > 0 and all(
        abs(c["rr_bpm"] - raw_rr_bpm) > 0.75 for c in kept
    ):
        raw_candidate = min(candidates, key=lambda c: abs(c["rr_bpm"] - raw_rr_bpm))
        if kept:
            kept[-1] = raw_candidate
        else:
            kept = [raw_candidate]

    return sorted(kept, key=lambda c: c["rr_bpm"])


def _transition_cost(
    previous_rr: float,
    current_rr: float,
    hop_s: float,
) -> tuple[float, bool]:
    hop_scale = max(float(hop_s), 1.0)
    cost = CONTINUITY_PENALTY_PER_BPM * abs(current_rr - previous_rr) / hop_scale
    harmonic_jump, _, _ = _near_integer_harmonic(previous_rr, current_rr)
    if harmonic_jump:
        cost += HARMONIC_JUMP_PENALTY / hop_scale
    return float(cost), bool(harmonic_jump)


def harmonic_viterbi_track(
    candidate_sets: list[list[dict[str, float]]],
    hop_s: float = 1.0,
) -> dict[str, np.ndarray]:
    """Select the most plausible RR trajectory across overlapping windows.

    Emission evidence is the local candidate score. Transition cost discourages
    implausibly fast changes and adds a larger penalty for exact 2x/3x jumps.
    The penalty is finite: a genuine, persistent rate change can accumulate
    enough local evidence to overcome it.
    """
    if not candidate_sets:
        empty = np.array([], dtype=float)
        return {
            "rr_bpm": empty,
            "selected_score": empty,
            "transition_cost": empty,
            "harmonic_transition": np.array([], dtype=bool),
            "path_index": np.array([], dtype=int),
        }
    if any(len(candidates) == 0 for candidates in candidate_sets):
        raise ValueError("Every temporal window must contain at least one candidate.")

    costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    transition_costs: list[np.ndarray] = []
    transition_harmonic: list[np.ndarray] = []

    first = candidate_sets[0]
    costs.append(np.array([-float(c["score"]) for c in first], dtype=float))
    backpointers.append(np.full(len(first), -1, dtype=int))
    transition_costs.append(np.zeros(len(first), dtype=float))
    transition_harmonic.append(np.zeros(len(first), dtype=bool))

    for index in range(1, len(candidate_sets)):
        previous = candidate_sets[index - 1]
        current = candidate_sets[index]
        current_cost = np.full(len(current), np.inf, dtype=float)
        current_back = np.full(len(current), -1, dtype=int)
        current_transition = np.full(len(current), np.nan, dtype=float)
        current_harmonic = np.zeros(len(current), dtype=bool)

        for j, candidate in enumerate(current):
            rr = float(candidate["rr_bpm"])
            emission_cost = -float(candidate["score"])
            for k, prev_candidate in enumerate(previous):
                prev_rr = float(prev_candidate["rr_bpm"])
                trans_cost, harmonic_jump = _transition_cost(prev_rr, rr, hop_s)
                total = costs[index - 1][k] + trans_cost + emission_cost
                if total < current_cost[j]:
                    current_cost[j] = total
                    current_back[j] = k
                    current_transition[j] = trans_cost
                    current_harmonic[j] = harmonic_jump

        costs.append(current_cost)
        backpointers.append(current_back)
        transition_costs.append(current_transition)
        transition_harmonic.append(current_harmonic)

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
    selected_transition = np.array(
        [transition_costs[i][path[i]] for i in range(len(path))],
        dtype=float,
    )
    selected_harmonic = np.array(
        [transition_harmonic[i][path[i]] for i in range(len(path))],
        dtype=bool,
    )
    return {
        "rr_bpm": rr,
        "selected_score": selected_score,
        "transition_cost": selected_transition,
        "harmonic_transition": selected_harmonic,
        "path_index": np.asarray(path, dtype=int),
    }


def _harmonic_relation_label(raw_rr: float, tracked_rr: float) -> str:
    if not (np.isfinite(raw_rr) and np.isfinite(tracked_rr) and raw_rr > 0 and tracked_rr > 0):
        return ""
    if abs(raw_rr - tracked_rr) <= 0.5:
        return ""
    is_harmonic, multiple, _ = _near_integer_harmonic(raw_rr, tracked_rr)
    if not is_harmonic:
        return "nonharmonic"
    if raw_rr > tracked_rr:
        return f"{multiple}x_to_fundamental"
    return f"fundamental_to_{multiple}x"


def _window_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }


def apply_temporal_tracking(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Post-process v2 sliding windows with harmonic-aware temporal tracking.

    Full-record estimation remains untouched. The existing local-window decision
    is preserved in rr_final_raw_bpm and the old Kalman output is copied to
    rr_smoothed_bpm_v2 before rr_smoothed_bpm becomes the v3 Kalman result.
    """
    metrics_path = output_dir / "metrics.json"
    windows_path = output_dir / "windowed_rr.csv"
    if not metrics_path.exists() or not windows_path.exists():
        raise FileNotFoundError("Temporal v3 requires metrics.json and windowed_rr.csv from analyze_recording().")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path)
    if windows.empty:
        metrics["temporal_tracker"] = {
            "version": TEMPORAL_TRACKER_VERSION,
            "applied": False,
            "reason": "no sliding windows",
        }
        metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
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
    starts = list(range(0, len(filtered) - win_n + 1, hop_n)) if len(filtered) >= win_n else []
    if len(starts) != len(windows):
        raise RuntimeError(
            f"Temporal v3 window mismatch: recomputed {len(starts)} windows but v2 wrote {len(windows)}."
        )

    candidate_sets: list[list[dict[str, float]]] = []
    for row_index, start in enumerate(starts):
        stop = start + win_n
        try:
            local_surrogate, local_eigvals, _, local_ratio = pca_surrogate(filtered[start:stop])
            local_move_q, _, _ = motion_quality(projected[start:stop], fs_hz, cfg)
            local_est, local_psd, local_acf = estimate_surrogate(
                local_surrogate,
                local_eigvals,
                local_ratio,
                fs_hz,
                cfg,
                timestamp_quality=ts_qc.timestamp_quality,
                movement_quality=local_move_q,
            )
            raw_rr = float(windows.loc[row_index, "rr_final_raw_bpm"])
            candidates = temporal_candidate_set(local_psd, local_acf, cfg, raw_rr)
        except Exception:
            raw_rr = float(windows.loc[row_index, "rr_final_raw_bpm"])
            confidence = float(windows.loc[row_index, "confidence"])
            fallback_score = max(0.05, min(1.0, confidence)) if np.isfinite(confidence) else 0.05
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

    track = harmonic_viterbi_track(candidate_sets, hop_s=cfg.hop_s)
    raw_rr = windows["rr_final_raw_bpm"].to_numpy(float)
    tracked_rr = track["rr_bpm"]

    if "rr_smoothed_bpm" in windows.columns:
        windows["rr_smoothed_bpm_v2"] = windows["rr_smoothed_bpm"]
    if "error_smoothed_bpm" in windows.columns:
        windows["error_smoothed_bpm_v2"] = windows["error_smoothed_bpm"]

    windows["rr_temporal_bpm"] = tracked_rr
    windows["temporal_selected_score"] = track["selected_score"]
    windows["temporal_transition_cost"] = track["transition_cost"]
    windows["temporal_harmonic_transition"] = track["harmonic_transition"]
    windows["temporal_candidate_count"] = [len(candidates) for candidates in candidate_sets]
    windows["temporal_changed_from_raw"] = np.abs(tracked_rr - raw_rr) > 0.5
    windows["temporal_relation"] = [
        _harmonic_relation_label(raw, tracked) for raw, tracked in zip(raw_rr, tracked_rr)
    ]

    confidence = windows["confidence"].to_numpy(float)
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

    valid_mask = windows["valid"].to_numpy(bool) if "valid" in windows.columns else np.ones(len(windows), bool)
    valid = windows.loc[valid_mask]
    ws = dict(metrics.get("window_stats", {}))

    # Preserve v2 Kalman statistics before replacing the generic smoothed fields.
    for key in (
        "rr_smoothed_mean_bpm",
        "rr_smoothed_median_bpm",
        "rr_smoothed_std_bpm",
        "mae_smoothed_bpm",
        "rmse_smoothed_bpm",
        "bias_smoothed_bpm",
    ):
        if key in ws and f"v2_{key}" not in ws:
            ws[f"v2_{key}"] = ws[key]

    temporal_values = valid["rr_temporal_bpm"].to_numpy(float)
    smoothed_values = valid["rr_smoothed_bpm"].to_numpy(float)
    temporal_stats = _window_stats(temporal_values)
    smoothed_stats = _window_stats(smoothed_values)

    ws.update(
        {
            "rr_temporal_mean_bpm": temporal_stats["mean"],
            "rr_temporal_median_bpm": temporal_stats["median"],
            "rr_temporal_std_bpm": temporal_stats["std"],
            "rr_smoothed_mean_bpm": smoothed_stats["mean"],
            "rr_smoothed_median_bpm": smoothed_stats["median"],
            "rr_smoothed_std_bpm": smoothed_stats["std"],
            "temporal_corrections_count": int(windows["temporal_changed_from_raw"].sum()),
            "temporal_correction_fraction": float(windows["temporal_changed_from_raw"].mean()),
            "temporal_harmonic_corrections_count": int(
                windows["temporal_relation"].isin(["2x_to_fundamental", "3x_to_fundamental"]).sum()
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
    metrics["temporal_tracker"] = {
        "version": TEMPORAL_TRACKER_VERSION,
        "applied": True,
        "continuity_penalty_per_bpm": CONTINUITY_PENALTY_PER_BPM,
        "harmonic_jump_penalty": HARMONIC_JUMP_PENALTY,
        "harmonic_ratio_tolerance": HARMONIC_RATIO_TOLERANCE,
        "soft_alias_bonus": SOFT_ALIAS_BONUS,
        "max_candidates_per_window": MAX_CANDIDATES_PER_WINDOW,
        "corrections_count": int(windows["temporal_changed_from_raw"].sum()),
        "harmonic_corrections_count": int(
            windows["temporal_relation"].isin(["2x_to_fundamental", "3x_to_fundamental"]).sum()
        ),
    }
    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_temporal_diagnostics(output_dir / "temporal_v3.png", windows, target_rpm, cfg)
    return metrics


def _plot_temporal_diagnostics(
    path: Path,
    windows: pd.DataFrame,
    target_rpm: float,
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    time_s = windows["time_s"]

    axes[0].plot(time_s, windows["rr_final_raw_bpm"], label="Local v2")
    axes[0].plot(time_s, windows["rr_temporal_bpm"], label="Temporal v3")
    if "rr_smoothed_bpm_v2" in windows:
        axes[0].plot(time_s, windows["rr_smoothed_bpm_v2"], label="Kalman v2", alpha=0.65)
    axes[0].plot(time_s, windows["rr_smoothed_bpm"], label="Temporal v3 + Kalman")
    if np.isfinite(target_rpm):
        axes[0].axhline(target_rpm, linestyle="--", label=f"Alvo {target_rpm:g}")
    changed = windows["temporal_changed_from_raw"].to_numpy(bool)
    if changed.any():
        axes[0].scatter(
            windows.loc[changed, "time_s"],
            windows.loc[changed, "rr_temporal_bpm"],
            marker="x",
            label="Correção temporal",
        )
    axes[0].set_ylabel("RR [resp/min]")
    axes[0].set_title(
        f"Temporal v3 — janelas {cfg.window_s:g}s / hop {cfg.hop_s:g}s"
    )
    axes[0].legend(loc="upper right", ncols=2)

    axes[1].plot(time_s, windows["confidence"], label="Confiança local")
    axes[1].plot(time_s, windows["temporal_selected_score"], label="Score do estado v3")
    axes[1].axhline(cfg.min_confidence, linestyle="--", label="Min. confiança")
    axes[1].set_xlabel("Tempo [s]")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Evidência local e estados selecionados")

    fig.savefig(path, dpi=160)
    plt.close(fig)
