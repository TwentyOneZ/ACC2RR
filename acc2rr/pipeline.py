from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    Config, _finite_float, _jsonable, butter_bandpass_zero_phase, estimate_gravity,
    hampel_vector_despike, load_metadata, pca_surrogate,
    project_perpendicular_to_gravity, read_acc_csv, resample_uniform,
    timestamp_qc_and_segment,
)
from .estimators import estimate_surrogate, kalman_track_rr, motion_quality

def analyze_recording(
    acc_path: Path,
    output_dir: Path,
    cfg: Config,
    save_intermediate: bool = False,
) -> dict[str, Any]:
    recording_dir = acc_path.parent
    metadata = load_metadata(recording_dir)
    nominal_fs = _finite_float(metadata.get("acc_sample_rate_hz"))
    nominal_fs_arg = nominal_fs if np.isfinite(nominal_fs) else None

    raw_df = read_acc_csv(acc_path)
    segment_df, ts_qc = timestamp_qc_and_segment(raw_df, nominal_fs_arg, cfg)
    fs_hz = ts_qc.nominal_fs_hz
    t, xyz_resampled, epoch0 = resample_uniform(segment_df, fs_hz)
    clean_xyz, outlier_mask = hampel_vector_despike(
        xyz_resampled, fs_hz, cfg.hampel_window_s, cfg.hampel_n_sigma
    )
    gravity_mg, gravity_unit = estimate_gravity(clean_xyz)
    projected = project_perpendicular_to_gravity(clean_xyz, gravity_mg, gravity_unit)
    filtered = butter_bandpass_zero_phase(
        projected, fs_hz, cfg.fmin_hz, cfg.fmax_hz, cfg.filter_order
    )
    surrogate, eigvals, eigvecs, pca_ratio = pca_surrogate(filtered)
    move_quality, move_snr_db, move_band_power = motion_quality(projected, fs_hz, cfg)
    estimate, psd, acf = estimate_surrogate(
        surrogate,
        eigvals,
        pca_ratio,
        fs_hz,
        cfg,
        timestamp_quality=ts_qc.timestamp_quality,
        movement_quality=move_quality,
    )

    position = recording_dir.parent.name
    condition = recording_dir.name
    target_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*rpm", condition, re.I)
    target_rpm = float(target_match.group(1)) if target_match else float("nan")

    # Sliding-window estimation is run on the already projected+filtered full
    # record; this avoids repeated filtfilt edge transients in every window.
    win_n = max(16, int(round(cfg.window_s * fs_hz)))
    hop_n = max(1, int(round(cfg.hop_s * fs_hz)))
    window_rows: list[dict[str, Any]] = []
    if len(filtered) >= win_n:
        starts = range(0, len(filtered) - win_n + 1, hop_n)
        for start in starts:
            stop = start + win_n
            local_surrogate, local_eigvals, _, local_ratio = pca_surrogate(filtered[start:stop])
            local_move_q, local_snr_db, _ = motion_quality(projected[start:stop], fs_hz, cfg)
            try:
                local_est, _, _ = estimate_surrogate(
                    local_surrogate,
                    local_eigvals,
                    local_ratio,
                    fs_hz,
                    cfg,
                    timestamp_quality=ts_qc.timestamp_quality,
                    movement_quality=local_move_q,
                )
                center_idx = start + win_n // 2
                row = {
                    "time_s": float(t[center_idx]),
                    "timestamp_epoch_s": float(epoch0 + t[center_idx]),
                    "rr_psd_raw_bpm": local_est.rr_psd_raw_bpm,
                    "rr_psd_bpm": local_est.rr_psd_bpm,
                    "rr_acf_bpm": local_est.rr_acf_bpm,
                    "rr_final_raw_bpm": local_est.rr_final_bpm,
                    "confidence": local_est.confidence,
                    "spectral_quality": local_est.spectral_quality,
                    "acf_quality": local_est.acf_quality,
                    "agreement_quality": local_est.agreement_quality,
                    "pca_pc1_variance_ratio": local_est.pca_pc1_variance_ratio,
                    "motion_quality": local_move_q,
                    "motion_snr_db": local_snr_db,
                    "valid": bool(local_est.confidence >= cfg.min_confidence),
                }
            except Exception as exc:  # preserve the rest of a long batch analysis
                center_idx = start + win_n // 2
                row = {
                    "time_s": float(t[center_idx]),
                    "timestamp_epoch_s": float(epoch0 + t[center_idx]),
                    "rr_psd_raw_bpm": np.nan,
                    "rr_psd_bpm": np.nan,
                    "rr_acf_bpm": np.nan,
                    "rr_final_raw_bpm": np.nan,
                    "confidence": 0.0,
                    "spectral_quality": 0.0,
                    "acf_quality": 0.0,
                    "agreement_quality": 0.0,
                    "pca_pc1_variance_ratio": np.nan,
                    "motion_quality": 0.0,
                    "motion_snr_db": np.nan,
                    "valid": False,
                    "error": str(exc),
                }
            window_rows.append(row)

    windows = pd.DataFrame(window_rows)
    if not windows.empty:
        windows["rr_smoothed_bpm"] = kalman_track_rr(
            windows["rr_final_raw_bpm"].to_numpy(float),
            windows["confidence"].to_numpy(float),
            cfg.hop_s,
            cfg,
        )
        if np.isfinite(target_rpm):
            windows["error_raw_bpm"] = windows["rr_final_raw_bpm"] - target_rpm
            windows["error_smoothed_bpm"] = windows["rr_smoothed_bpm"] - target_rpm

    output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(output_dir / "windowed_rr.csv", index=False)

    valid_windows = windows[windows["valid"]] if not windows.empty else windows
    if not valid_windows.empty:
        valid_rr = valid_windows["rr_final_raw_bpm"].to_numpy(float)
        valid_smoothed = valid_windows["rr_smoothed_bpm"].to_numpy(float)
        window_stats = {
            "n_windows": int(len(windows)),
            "n_valid_windows": int(len(valid_windows)),
            "valid_fraction": float(len(valid_windows) / len(windows)),
            "rr_raw_mean_bpm": float(np.nanmean(valid_rr)),
            "rr_raw_median_bpm": float(np.nanmedian(valid_rr)),
            "rr_raw_std_bpm": float(np.nanstd(valid_rr)),
            "rr_smoothed_mean_bpm": float(np.nanmean(valid_smoothed)),
            "rr_smoothed_median_bpm": float(np.nanmedian(valid_smoothed)),
            "rr_smoothed_std_bpm": float(np.nanstd(valid_smoothed)),
        }
        if np.isfinite(target_rpm):
            raw_err = valid_rr - target_rpm
            smooth_err = valid_smoothed - target_rpm
            window_stats.update(
                {
                    "mae_raw_bpm": float(np.nanmean(np.abs(raw_err))),
                    "rmse_raw_bpm": float(np.sqrt(np.nanmean(raw_err**2))),
                    "bias_raw_bpm": float(np.nanmean(raw_err)),
                    "mae_smoothed_bpm": float(np.nanmean(np.abs(smooth_err))),
                    "rmse_smoothed_bpm": float(np.sqrt(np.nanmean(smooth_err**2))),
                    "bias_smoothed_bpm": float(np.nanmean(smooth_err)),
                }
            )
    else:
        window_stats = {
            "n_windows": int(len(windows)),
            "n_valid_windows": 0,
            "valid_fraction": 0.0,
        }

    metrics: dict[str, Any] = {
        "source": str(acc_path),
        "position": position,
        "condition": condition,
        "target_rpm": target_rpm,
        "config": asdict(cfg),
        "metadata": metadata,
        "timestamp_qc": asdict(ts_qc),
        "resampled_samples": int(len(t)),
        "analysis_duration_s": float(t[-1] - t[0]),
        "fs_hz": fs_hz,
        "hampel_outliers": int(outlier_mask.sum()),
        "hampel_outlier_fraction": float(outlier_mask.mean()),
        "gravity_vector_mg": gravity_mg.tolist(),
        "gravity_unit_vector": gravity_unit.tolist(),
        "gravity_magnitude_mg": float(np.linalg.norm(gravity_mg)),
        "pca_components_columns": eigvecs.tolist(),
        "full_record_estimate": asdict(estimate),
        "motion_quality": move_quality,
        "motion_snr_db": move_snr_db,
        "motion_band_power": move_band_power,
        "window_stats": window_stats,
    }
    if np.isfinite(target_rpm):
        metrics["full_record_error_bpm"] = estimate.rr_final_bpm - target_rpm
        metrics["full_record_absolute_error_bpm"] = abs(estimate.rr_final_bpm - target_rpm)

    (output_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if save_intermediate:
        intermediate = pd.DataFrame(
            {
                "time_s": t,
                "timestamp_epoch_s": epoch0 + t,
                "x_resampled_mg": xyz_resampled[:, 0],
                "y_resampled_mg": xyz_resampled[:, 1],
                "z_resampled_mg": xyz_resampled[:, 2],
                "hampel_outlier": outlier_mask,
                "x_clean_mg": clean_xyz[:, 0],
                "y_clean_mg": clean_xyz[:, 1],
                "z_clean_mg": clean_xyz[:, 2],
                "x_projected_mg": projected[:, 0],
                "y_projected_mg": projected[:, 1],
                "z_projected_mg": projected[:, 2],
                "x_band_mg": filtered[:, 0],
                "y_band_mg": filtered[:, 1],
                "z_band_mg": filtered[:, 2],
                "respiratory_surrogate": surrogate,
            }
        )
        intermediate.to_csv(output_dir / "intermediate_signals.csv", index=False)

    plot_diagnostics(
        output_dir / "diagnostics.png",
        t,
        xyz_resampled,
        filtered,
        surrogate,
        psd,
        acf,
        estimate,
        windows,
        target_rpm,
        position,
        condition,
        cfg,
    )
    return metrics


def plot_diagnostics(
    path: Path,
    t: np.ndarray,
    xyz_raw: np.ndarray,
    filtered: np.ndarray,
    surrogate: np.ndarray,
    psd: dict[str, Any],
    acf: dict[str, Any],
    estimate: Estimate,
    windows: pd.DataFrame,
    target_rpm: float,
    position: str,
    condition: str,
    cfg: Config,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13, 18), constrained_layout=True)

    axes[0].plot(t, xyz_raw[:, 0], label="x")
    axes[0].plot(t, xyz_raw[:, 1], label="y")
    axes[0].plot(t, xyz_raw[:, 2], label="z")
    axes[0].set_ylabel("Aceleração [mg]")
    axes[0].set_title(f"{position} / {condition} — XYZ reamostrado")
    axes[0].legend(loc="upper right", ncols=3)

    axes[1].plot(t, filtered[:, 0], label="x_perp")
    axes[1].plot(t, filtered[:, 1], label="y_perp")
    axes[1].plot(t, filtered[:, 2], label="z_perp")
    axes[1].set_ylabel("Banda resp. [mg]")
    axes[1].set_title(f"Projeção perpendicular à gravidade + Butterworth {cfg.fmin_hz:.2f}–{cfg.fmax_hz:.2f} Hz")
    axes[1].legend(loc="upper right", ncols=3)

    axes[2].plot(t, surrogate)
    axes[2].set_ylabel("PC1 [a.u.]")
    axes[2].set_title(
        f"Respiratory surrogate — PC1 explica {100 * estimate.pca_pc1_variance_ratio:.1f}% da variância"
    )

    band = psd["band_mask"]
    axes[3].plot(psd["freq_hz"][band] * 60.0, psd["power"][band])
    axes[3].axvline(estimate.rr_psd_raw_bpm, linestyle="--", label=f"PSD bruto {estimate.rr_psd_raw_bpm:.2f}")
    axes[3].axvline(estimate.rr_psd_bpm, linestyle=":", label=f"PSD usado {estimate.rr_psd_bpm:.2f}")
    if np.isfinite(target_rpm):
        axes[3].axvline(target_rpm, linestyle="-.", label=f"Alvo {target_rpm:g}")
    axes[3].set_xlabel("Respirações/min")
    axes[3].set_ylabel("PSD")
    axes[3].set_title(f"Welch PSD — qualidade={estimate.spectral_quality:.3f}")
    axes[3].legend(loc="upper right")

    acf_mask = (acf["lags_s"] >= 1.0 / cfg.fmax_hz) & (acf["lags_s"] <= 1.0 / cfg.fmin_hz)
    axes[4].plot(acf["lags_s"][acf_mask], acf["acf"][acf_mask])
    if np.isfinite(acf.get("selected_lag_s", np.nan)):
        axes[4].axvline(acf["selected_lag_s"], linestyle="--", label=f"ACF {estimate.rr_acf_bpm:.2f} rpm")
    axes[4].set_xlabel("Lag [s]")
    axes[4].set_ylabel("Autocorrelação")
    axes[4].set_title(f"Autocorrelação — qualidade={estimate.acf_quality:.3f}")
    axes[4].legend(loc="upper right")

    if not windows.empty:
        axes[5].plot(windows["time_s"], windows["rr_final_raw_bpm"], label="RR janela")
        axes[5].plot(windows["time_s"], windows["rr_smoothed_bpm"], label="RR rastreado")
        if np.isfinite(target_rpm):
            axes[5].axhline(target_rpm, linestyle="--", label=f"Alvo {target_rpm:g}")
        invalid = ~windows["valid"].to_numpy(bool)
        if invalid.any():
            axes[5].scatter(
                windows.loc[invalid, "time_s"],
                windows.loc[invalid, "rr_final_raw_bpm"],
                marker="x",
                label="Baixa confiança",
            )
    axes[5].set_xlabel("Tempo [s]")
    axes[5].set_ylabel("RR [resp/min]")
    axes[5].set_title(
        f"Janelas {cfg.window_s:g}s / hop {cfg.hop_s:g}s — RR completo={estimate.rr_final_bpm:.2f}, confiança={estimate.confidence:.3f}"
    )
    axes[5].legend(loc="upper right")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def discover_acc_files(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("**/acc.csv"))


def flatten_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    est = metrics["full_record_estimate"]
    ts = metrics["timestamp_qc"]
    ws = metrics["window_stats"]
    return {
        "position": metrics["position"],
        "condition": metrics["condition"],
        "target_rpm": metrics.get("target_rpm"),
        "duration_s": metrics["analysis_duration_s"],
        "fs_hz": metrics["fs_hz"],
        "effective_fs_hz": ts["effective_fs_hz"],
        "timestamp_jitter_ms": ts["dt_jitter_ms"],
        "long_gaps": ts["long_gap_count"],
        "estimated_missing_samples": ts["estimated_missing_samples"],
        "timestamp_quality": ts["timestamp_quality"],
        "hampel_outliers_pct": 100.0 * metrics["hampel_outlier_fraction"],
        "gravity_magnitude_mg": metrics["gravity_magnitude_mg"],
        "pca_pc1_variance_ratio": est["pca_pc1_variance_ratio"],
        "rr_psd_raw_bpm": est["rr_psd_raw_bpm"],
        "rr_psd_bpm": est["rr_psd_bpm"],
        "rr_acf_bpm": est["rr_acf_bpm"],
        "rr_final_bpm": est["rr_final_bpm"],
        "error_bpm": metrics.get("full_record_error_bpm"),
        "abs_error_bpm": metrics.get("full_record_absolute_error_bpm"),
        "spectral_quality": est["spectral_quality"],
        "acf_quality": est["acf_quality"],
        "agreement_quality": est["agreement_quality"],
        "motion_quality": metrics["motion_quality"],
        "motion_snr_db": metrics["motion_snr_db"],
        "confidence": est["confidence"],
        "n_windows": ws.get("n_windows"),
        "valid_window_fraction": ws.get("valid_fraction"),
        "window_rr_median_bpm": ws.get("rr_raw_median_bpm"),
        "window_rr_smoothed_median_bpm": ws.get("rr_smoothed_median_bpm"),
        "window_mae_raw_bpm": ws.get("mae_raw_bpm"),
        "window_mae_smoothed_bpm": ws.get("mae_smoothed_bpm"),
    }


def write_summary_report(summary: pd.DataFrame, output_root: Path, cfg: Config) -> None:
    summary.to_csv(output_root / "summary.csv", index=False)

    lines = [
        "# ACC2RR — relatório do baseline",
        "",
        "Pipeline: timestamp QC → reamostragem → Hampel vetorial → referência pela gravidade → "
        "projeção perpendicular → Butterworth zero-phase → PCA → PSD + autocorrelação → consenso → janelas móveis.",
        "",
        f"Banda respiratória: **{cfg.fmin_hz:.2f}–{cfg.fmax_hz:.2f} Hz**; janela móvel: **{cfg.window_s:g} s**; hop: **{cfg.hop_s:g} s**.",
        "",
    ]

    display_cols = [
        "position",
        "condition",
        "target_rpm",
        "rr_psd_bpm",
        "rr_acf_bpm",
        "rr_final_bpm",
        "abs_error_bpm",
        "confidence",
        "pca_pc1_variance_ratio",
        "valid_window_fraction",
        "window_mae_smoothed_bpm",
    ]
    existing = [c for c in display_cols if c in summary.columns]
    if not summary.empty:
        lines.append(summary[existing].round(4).to_markdown(index=False))
        lines.append("")

        if "abs_error_bpm" in summary and summary["abs_error_bpm"].notna().any():
            mae = float(summary["abs_error_bpm"].mean())
            rmse = float(np.sqrt(np.nanmean(np.square(summary["error_bpm"].to_numpy(float)))))
            lines.extend(
                [
                    "## Métricas globais (coleta completa)",
                    "",
                    f"- MAE: **{mae:.3f} resp/min**",
                    f"- RMSE: **{rmse:.3f} resp/min**",
                    f"- Confiança média: **{summary['confidence'].mean():.3f}**",
                    "",
                ]
            )
    lines.extend(
        [
            "## Como interpretar",
            "",
            "- `rr_psd_bpm` e `rr_acf_bpm` são estimativas independentes; grande desacordo indica ambiguidade/harmônico/artefato.",
            "- `pca_pc1_variance_ratio` mede quanto da energia respiratória triaxial ficou concentrada na primeira componente.",
            "- `confidence` é um índice heurístico, não uma probabilidade calibrada; use-o para diagnóstico/comparação entre versões.",
            "- `windowed_rr.csv` em cada coleta contém a série temporal, confiança e flags de validade por janela.",
            "- `diagnostics.png` reúne sinais, espectro, autocorrelação e RR no tempo para inspeção visual.",
            "",
        ]
    )
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
