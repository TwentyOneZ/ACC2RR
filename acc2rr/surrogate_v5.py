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
from .estimators import autocorrelation_estimate, consensus_estimate, psd_estimate

SURROGATE_BENCHMARK_VERSION = "surrogate_benchmark_v1"
PC12_ANGLE_STEP_DEG = 5
PRIMITIVE_SURROGATES = (
    "pc1",
    "pc2",
    "pc3",
    "axis_x",
    "axis_y",
    "axis_z",
    "vector_magnitude",
)
DISPLAY_SURROGATES = (
    "pc1",
    "pc2",
    "best_axis",
    "best_pca_component",
    "optimized_pc12",
)


def _normalized_psd_at(f_hz: float, psd: dict[str, Any]) -> float:
    if not np.isfinite(f_hz) or f_hz <= 0:
        return 0.0
    p = float(np.interp(f_hz, psd["freq_hz"], psd["power"], left=0.0, right=0.0))
    return float(np.clip(p / max(float(psd.get("p_max", 0.0)), EPS), 0.0, 1.0))


def _degenerate_result(reference_rr_bpm: float) -> dict[str, Any]:
    return {
        "rr_psd_raw_bpm": float("nan"),
        "rr_psd_bpm": float("nan"),
        "rr_acf_bpm": float("nan"),
        "rr_consensus_bpm": float("nan"),
        "spectral_quality": 0.0,
        "acf_quality": 0.0,
        "agreement_quality": 0.0,
        "consensus_score": 0.0,
        "benchmark_confidence": 0.0,
        "self_fundamental_power": 0.0,
        "self_second_harmonic_power": 0.0,
        "self_fundamental_to_2x_ratio": float("nan"),
        "reference_rr_bpm": reference_rr_bpm,
        "reference_fundamental_power": 0.0,
        "reference_second_harmonic_power": 0.0,
        "reference_fundamental_to_2x_ratio": float("nan"),
    }


def evaluate_surrogate(
    surrogate: np.ndarray,
    fs_hz: float,
    cfg: Config,
    reference_rr_bpm: float = float("nan"),
) -> dict[str, Any]:
    """Evaluate one 1-D respiratory surrogate without using an experimental target.

    reference_rr_bpm is optional diagnostic context derived from the full-record
    ACC estimate. It is used only to report power around that data-derived
    frequency and its second harmonic; it does not affect RR estimation or score.
    """
    x = np.asarray(surrogate, dtype=float)
    if len(x) < 16 or not np.isfinite(x).all() or float(np.std(x)) < 1e-10:
        return _degenerate_result(reference_rr_bpm)

    psd = psd_estimate(x, fs_hz, cfg)
    acf = autocorrelation_estimate(x, fs_hz, cfg)
    rr_consensus, consensus_score, agreement = consensus_estimate(psd, acf, cfg)

    spectral_quality = float(psd.get("quality", 0.0))
    acf_quality = float(acf.get("quality", 0.0))
    benchmark_confidence = float(
        np.clip(
            0.35 * spectral_quality
            + 0.30 * acf_quality
            + 0.20 * agreement
            + 0.15 * consensus_score,
            0.0,
            1.0,
        )
    )

    self_f = rr_consensus / 60.0 if np.isfinite(rr_consensus) else float("nan")
    self_direct = _normalized_psd_at(self_f, psd)
    self_second = (
        _normalized_psd_at(2.0 * self_f, psd)
        if np.isfinite(self_f) and 2.0 * self_f <= cfg.fmax_hz
        else 0.0
    )
    self_ratio = (
        self_direct / max(self_second, EPS)
        if np.isfinite(self_f) and self_second > EPS
        else float("inf") if np.isfinite(self_f) and self_direct > 0 else float("nan")
    )

    reference_f = (
        reference_rr_bpm / 60.0
        if np.isfinite(reference_rr_bpm) and reference_rr_bpm > 0
        else float("nan")
    )
    reference_direct = _normalized_psd_at(reference_f, psd)
    reference_second = (
        _normalized_psd_at(2.0 * reference_f, psd)
        if np.isfinite(reference_f) and 2.0 * reference_f <= cfg.fmax_hz
        else 0.0
    )
    reference_ratio = (
        reference_direct / max(reference_second, EPS)
        if np.isfinite(reference_f) and reference_second > EPS
        else float("inf")
        if np.isfinite(reference_f) and reference_direct > 0
        else float("nan")
    )

    return {
        "rr_psd_raw_bpm": float(psd.get("rr_raw_bpm", float("nan"))),
        "rr_psd_bpm": float(psd.get("rr_bpm", float("nan"))),
        "rr_acf_bpm": float(acf.get("rr_bpm", float("nan"))),
        "rr_consensus_bpm": float(rr_consensus),
        "spectral_quality": spectral_quality,
        "acf_quality": acf_quality,
        "agreement_quality": float(agreement),
        "consensus_score": float(consensus_score),
        "benchmark_confidence": benchmark_confidence,
        "self_fundamental_power": self_direct,
        "self_second_harmonic_power": self_second,
        "self_fundamental_to_2x_ratio": float(self_ratio),
        "reference_rr_bpm": float(reference_rr_bpm),
        "reference_fundamental_power": reference_direct,
        "reference_second_harmonic_power": reference_second,
        "reference_fundamental_to_2x_ratio": float(reference_ratio),
    }


def _window_pca(filtered_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = filtered_window - np.mean(filtered_window, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    components = centered @ eigvecs
    total = float(np.sum(eigvals))
    ratios = eigvals / total if total > EPS else np.zeros_like(eigvals)
    return components, eigvals, ratios


def _record(
    surrogate_name: str,
    evaluation: dict[str, Any],
    *,
    pca_explained_variance_ratio: float = float("nan"),
    source_surrogate: str = "",
    angle_deg: float = float("nan"),
) -> dict[str, Any]:
    row = {
        "surrogate": surrogate_name,
        "source_surrogate": source_surrogate,
        "angle_deg": angle_deg,
        "pca_explained_variance_ratio": pca_explained_variance_ratio,
    }
    row.update(evaluation)
    return row


def benchmark_window(
    filtered_window: np.ndarray,
    fs_hz: float,
    cfg: Config,
    reference_rr_bpm: float = float("nan"),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate primitive and selected surrogate representations for one window."""
    components, _, ratios = _window_pca(filtered_window)

    records: list[dict[str, Any]] = []
    primitive_by_name: dict[str, dict[str, Any]] = {}

    for index in range(3):
        name = f"pc{index + 1}"
        row = _record(
            name,
            evaluate_surrogate(components[:, index], fs_hz, cfg, reference_rr_bpm),
            pca_explained_variance_ratio=float(ratios[index]),
        )
        records.append(row)
        primitive_by_name[name] = row

    for index, axis_name in enumerate(("axis_x", "axis_y", "axis_z")):
        row = _record(
            axis_name,
            evaluate_surrogate(filtered_window[:, index], fs_hz, cfg, reference_rr_bpm),
        )
        records.append(row)
        primitive_by_name[axis_name] = row

    magnitude = np.linalg.norm(filtered_window, axis=1)
    mag_row = _record(
        "vector_magnitude",
        evaluate_surrogate(magnitude, fs_hz, cfg, reference_rr_bpm),
    )
    records.append(mag_row)
    primitive_by_name["vector_magnitude"] = mag_row

    best_axis_source = max(
        (primitive_by_name[name] for name in ("axis_x", "axis_y", "axis_z")),
        key=lambda row: float(row["benchmark_confidence"]),
    )
    best_axis = dict(best_axis_source)
    best_axis["surrogate"] = "best_axis"
    best_axis["source_surrogate"] = str(best_axis_source["surrogate"])
    records.append(best_axis)

    best_pca_source = max(
        (primitive_by_name[name] for name in ("pc1", "pc2", "pc3")),
        key=lambda row: float(row["benchmark_confidence"]),
    )
    best_pca = dict(best_pca_source)
    best_pca["surrogate"] = "best_pca_component"
    best_pca["source_surrogate"] = str(best_pca_source["surrogate"])
    records.append(best_pca)

    angle_records: list[dict[str, Any]] = []
    pc1 = components[:, 0]
    pc2 = components[:, 1]
    for angle_deg in range(0, 180, PC12_ANGLE_STEP_DEG):
        theta = math.radians(angle_deg)
        surrogate = pc1 * math.cos(theta) + pc2 * math.sin(theta)
        evaluation = evaluate_surrogate(surrogate, fs_hz, cfg, reference_rr_bpm)
        angle_records.append(
            _record(
                "pc12_angle",
                evaluation,
                angle_deg=float(angle_deg),
            )
        )

    best_angle = max(
        angle_records,
        key=lambda row: float(row["benchmark_confidence"]),
    )
    optimized = dict(best_angle)
    optimized["surrogate"] = "optimized_pc12"
    optimized["source_surrogate"] = "pc1_pc2_scan"
    records.append(optimized)

    return records, angle_records


def _summarize_surrogates(df: pd.DataFrame, target_rpm: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surrogate, group in df.groupby("surrogate", sort=False):
        rr = group["rr_consensus_bpm"].to_numpy(float)
        finite_rr = rr[np.isfinite(rr)]
        ratios = group["reference_fundamental_to_2x_ratio"].to_numpy(float)
        finite_ratios = ratios[np.isfinite(ratios)]
        row: dict[str, Any] = {
            "surrogate": surrogate,
            "n_windows": int(len(group)),
            "n_finite_rr": int(len(finite_rr)),
            "rr_mean_bpm": float(np.mean(finite_rr)) if len(finite_rr) else float("nan"),
            "rr_median_bpm": float(np.median(finite_rr)) if len(finite_rr) else float("nan"),
            "rr_std_bpm": float(np.std(finite_rr)) if len(finite_rr) else float("nan"),
            "benchmark_confidence_mean": float(group["benchmark_confidence"].mean()),
            "benchmark_confidence_median": float(group["benchmark_confidence"].median()),
            "spectral_quality_mean": float(group["spectral_quality"].mean()),
            "acf_quality_mean": float(group["acf_quality"].mean()),
            "agreement_quality_mean": float(group["agreement_quality"].mean()),
            "consensus_score_mean": float(group["consensus_score"].mean()),
            "reference_ratio_median": (
                float(np.median(finite_ratios)) if len(finite_ratios) else float("nan")
            ),
            "reference_fundamental_dominant_fraction": (
                float(np.mean(finite_ratios > 1.0)) if len(finite_ratios) else float("nan")
            ),
            "reference_ratio_ge_0_5_fraction": (
                float(np.mean(finite_ratios >= 0.5)) if len(finite_ratios) else float("nan")
            ),
            "reference_ratio_ge_0_2_fraction": (
                float(np.mean(finite_ratios >= 0.2)) if len(finite_ratios) else float("nan")
            ),
        }
        if np.isfinite(target_rpm) and len(finite_rr):
            err = finite_rr - target_rpm
            row.update(
                {
                    "target_mae_bpm": float(np.mean(np.abs(err))),
                    "target_rmse_bpm": float(np.sqrt(np.mean(err**2))),
                    "target_bias_bpm": float(np.mean(err)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_surrogate_benchmark_v5(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
) -> dict[str, Any]:
    """Run the v5 surrogate benchmark without changing the v4 RR output."""
    metrics_path = output_dir / "metrics.json"
    windows_path = output_dir / "windowed_rr.csv"
    if not metrics_path.exists() or not windows_path.exists():
        raise FileNotFoundError(
            "Surrogate v5 requires metrics.json and windowed_rr.csv from the v4 pipeline."
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    windows = pd.read_csv(windows_path)
    if windows.empty:
        metrics["surrogate_benchmark_v5"] = {
            "version": SURROGATE_BENCHMARK_VERSION,
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
    starts = list(range(0, len(filtered) - win_n + 1, hop_n)) if len(filtered) >= win_n else []
    if len(starts) != len(windows):
        raise RuntimeError(
            f"Surrogate v5 window mismatch: recomputed {len(starts)} windows "
            f"but pipeline wrote {len(windows)}."
        )

    full_est = metrics.get("full_record_estimate", {})
    reference_rr_bpm = float(full_est.get("rr_final_bpm", float("nan")))
    target_rpm = float(metrics.get("target_rpm", float("nan")))

    long_rows: list[dict[str, Any]] = []
    angle_rows: list[dict[str, Any]] = []

    for row_index, start in enumerate(starts):
        stop = start + win_n
        records, angles = benchmark_window(
            filtered[start:stop],
            fs_hz,
            cfg,
            reference_rr_bpm=reference_rr_bpm,
        )
        base = {
            "window_index": int(row_index),
            "time_s": float(windows.loc[row_index, "time_s"]),
            "timestamp_epoch_s": float(windows.loc[row_index, "timestamp_epoch_s"]),
            "local_v2_rr_bpm": float(windows.loc[row_index, "rr_final_raw_bpm"]),
            "v4_rr_bpm": float(windows.loc[row_index, "rr_temporal_bpm"])
            if "rr_temporal_bpm" in windows.columns
            else float("nan"),
            "local_confidence": float(windows.loc[row_index, "confidence"]),
        }

        for record in records:
            row = dict(base)
            row.update(record)
            if np.isfinite(target_rpm) and np.isfinite(float(row["rr_consensus_bpm"])):
                row["target_error_bpm"] = float(row["rr_consensus_bpm"]) - target_rpm
                row["target_abs_error_bpm"] = abs(float(row["target_error_bpm"]))
            long_rows.append(row)

        for record in angles:
            row = dict(base)
            row.update(record)
            angle_rows.append(row)

    benchmark = pd.DataFrame(long_rows)
    angle_scan = pd.DataFrame(angle_rows)
    summary = _summarize_surrogates(benchmark, target_rpm)

    benchmark.to_csv(output_dir / "surrogate_v5.csv", index=False)
    angle_scan.to_csv(output_dir / "surrogate_v5_angle_scan.csv", index=False)
    summary.to_csv(output_dir / "surrogate_v5_summary.csv", index=False)

    summary_records = json.loads(summary.to_json(orient="records"))
    metrics["surrogate_benchmark_v5"] = {
        "version": SURROGATE_BENCHMARK_VERSION,
        "applied": True,
        "selection_uses_target_rpm": False,
        "selection_rule": (
            "benchmark_confidence = 0.35*PSD_quality + 0.30*ACF_quality + "
            "0.20*PSD_ACF_agreement + 0.15*consensus_score"
        ),
        "reference_rr_source": "full_record_estimate.rr_final_bpm",
        "reference_rr_bpm": reference_rr_bpm,
        "angle_step_deg": PC12_ANGLE_STEP_DEG,
        "primitive_surrogates": list(PRIMITIVE_SURROGATES),
        "derived_surrogates": [
            "best_axis",
            "best_pca_component",
            "optimized_pc12",
        ],
        "summary": summary_records,
    }
    metrics_path.write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_surrogate_benchmark(
        output_dir / "surrogate_v5.png",
        benchmark,
        target_rpm,
        reference_rr_bpm,
    )
    return metrics


def _plot_surrogate_benchmark(
    path: Path,
    benchmark: pd.DataFrame,
    target_rpm: float,
    reference_rr_bpm: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    for surrogate in DISPLAY_SURROGATES:
        group = benchmark[benchmark["surrogate"] == surrogate]
        axes[0].plot(group["time_s"], group["rr_consensus_bpm"], label=surrogate)
    if np.isfinite(reference_rr_bpm):
        axes[0].axhline(reference_rr_bpm, linestyle=":", label="full-record RR")
    if np.isfinite(target_rpm):
        axes[0].axhline(target_rpm, linestyle="--", label="experimental target")
    axes[0].set_ylabel("RR [resp/min]")
    axes[0].set_title("Surrogate v5 — consensus RR by representation")
    axes[0].legend(loc="upper right", ncols=2)

    for surrogate in DISPLAY_SURROGATES:
        group = benchmark[benchmark["surrogate"] == surrogate]
        axes[1].plot(
            group["time_s"],
            group["reference_fundamental_to_2x_ratio"],
            label=surrogate,
        )
    axes[1].axhline(1.0, linestyle="--", label="fundamental = 2nd harmonic")
    axes[1].set_ylabel("P(f_ref) / P(2 f_ref)")
    axes[1].set_title("Data-derived full-record fundamental vs second harmonic")
    axes[1].legend(loc="upper right", ncols=2)

    for surrogate in DISPLAY_SURROGATES:
        group = benchmark[benchmark["surrogate"] == surrogate]
        axes[2].plot(group["time_s"], group["benchmark_confidence"], label=surrogate)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Benchmark confidence")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Internal respiratory evidence (target-free selection score)")
    axes[2].legend(loc="upper right", ncols=2)

    fig.savefig(path, dpi=160)
    plt.close(fig)
