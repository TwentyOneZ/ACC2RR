from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Any

import pandas as pd

from .core import Config
from .pipeline import analyze_recording, discover_acc_files, flatten_summary, write_summary_report
from .surrogate_v5 import apply_surrogate_benchmark_v5
from .temporal import apply_temporal_tracking
from .temporal_v4 import apply_temporal_tracking_v4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate respiratory rate from Polar H10 acc.csv recordings.")
    p.add_argument("--data-root", type=Path, default=Path("data"), help="Root containing */*/acc.csv recordings.")
    p.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for reports and plots.")
    p.add_argument("--recording", type=Path, help="Analyze one acc.csv or one directory containing acc.csv.")
    p.add_argument("--fmin", type=float, default=0.08, help="Minimum respiratory frequency [Hz].")
    p.add_argument("--fmax", type=float, default=0.80, help="Maximum respiratory frequency [Hz].")
    p.add_argument("--filter-order", type=int, default=4)
    p.add_argument("--hampel-window-s", type=float, default=0.40)
    p.add_argument("--hampel-sigma", type=float, default=4.0)
    p.add_argument("--window-s", type=float, default=30.0)
    p.add_argument("--hop-s", type=float, default=1.0)
    p.add_argument("--min-confidence", type=float, default=0.45)
    p.add_argument("--save-intermediate", action="store_true", help="Also write all preprocessed sample-level signals.")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config(
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        filter_order=args.filter_order,
        hampel_window_s=args.hampel_window_s,
        hampel_n_sigma=args.hampel_sigma,
        window_s=args.window_s,
        hop_s=args.hop_s,
        min_confidence=args.min_confidence,
    )

    if args.recording:
        rec = args.recording
        acc_files = [rec / "acc.csv"] if rec.is_dir() else [rec]
    else:
        acc_files = discover_acc_files(args.data_root)
    if not acc_files:
        raise SystemExit(f"No acc.csv files found under {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )

    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    print(f"Found {len(acc_files)} recording(s).")
    for acc_path in acc_files:
        try:
            if args.recording:
                try:
                    relative = acc_path.parent.relative_to(args.data_root)
                except ValueError:
                    relative = Path(acc_path.parent.parent.name) / acc_path.parent.name
            else:
                relative = acc_path.parent.relative_to(args.data_root)
            out = args.output_dir / relative
            print(f"[ACC2RR] {acc_path} -> {out}")
            metrics = analyze_recording(acc_path, out, cfg, save_intermediate=args.save_intermediate)
            metrics = apply_temporal_tracking(acc_path, out, cfg)
            metrics = apply_temporal_tracking_v4(acc_path, out, cfg)
            metrics = apply_surrogate_benchmark_v5(acc_path, out, cfg)

            summary_row = flatten_summary(metrics)
            ws = metrics.get("window_stats", {})
            summary_row.update(
                {
                    "window_rr_temporal_median_bpm": ws.get("rr_temporal_median_bpm"),
                    "window_mae_temporal_bpm": ws.get("mae_temporal_bpm"),
                    "window_rmse_temporal_bpm": ws.get("rmse_temporal_bpm"),
                    "window_bias_temporal_bpm": ws.get("bias_temporal_bpm"),
                    "window_temporal_corrections": ws.get("temporal_corrections_count"),
                    "window_temporal_harmonic_corrections": ws.get("temporal_harmonic_corrections_count"),
                    "window_v2_mae_smoothed_bpm": ws.get("v2_mae_smoothed_bpm"),
                    "window_v2_rmse_smoothed_bpm": ws.get("v2_rmse_smoothed_bpm"),
                    "window_v3_mae_temporal_bpm": ws.get("v3_mae_temporal_bpm"),
                    "window_v3_rmse_temporal_bpm": ws.get("v3_rmse_temporal_bpm"),
                    "window_v3_mae_smoothed_bpm": ws.get("v3_mae_smoothed_bpm"),
                    "window_v3_rmse_smoothed_bpm": ws.get("v3_rmse_smoothed_bpm"),
                }
            )

            v5_rows = metrics.get("surrogate_benchmark_v5", {}).get("summary", [])
            v5_by_name = {str(row.get("surrogate")): row for row in v5_rows}
            for name in ("pc1", "pc2", "best_axis", "best_pca_component", "optimized_pc12"):
                row = v5_by_name.get(name, {})
                prefix = f"v5_{name}"
                summary_row[f"{prefix}_target_mae_bpm"] = row.get("target_mae_bpm")
                summary_row[f"{prefix}_target_rmse_bpm"] = row.get("target_rmse_bpm")
                summary_row[f"{prefix}_confidence_mean"] = row.get("benchmark_confidence_mean")
                summary_row[f"{prefix}_reference_ratio_median"] = row.get("reference_ratio_median")
                summary_row[f"{prefix}_reference_fundamental_dominant_fraction"] = row.get(
                    "reference_fundamental_dominant_fraction"
                )

            summaries.append(summary_row)

            est = metrics["full_record_estimate"]
            print(
                f"  RR={est['rr_final_bpm']:.3f} bpm | PSD={est['rr_psd_bpm']:.3f} | "
                f"ACF={est['rr_acf_bpm']:.3f} | confidence={est['confidence']:.3f}"
            )
            if ws.get("rr_temporal_median_bpm") is not None:
                print(
                    f"  windows: v4 temporal median={ws['rr_temporal_median_bpm']:.3f} bpm | "
                    f"v4 Kalman median={ws.get('rr_smoothed_median_bpm', float('nan')):.3f} | "
                    f"corrections={ws.get('temporal_corrections_count', 0)}"
                )
            if v5_by_name:
                pc1 = v5_by_name.get("pc1", {})
                pc2 = v5_by_name.get("pc2", {})
                optimized = v5_by_name.get("optimized_pc12", {})
                print(
                    "  v5 benchmark: "
                    f"PC1 MAE={pc1.get('target_mae_bpm', float('nan')):.3f} | "
                    f"PC2 MAE={pc2.get('target_mae_bpm', float('nan')):.3f} | "
                    f"optimized PC1/PC2 MAE={optimized.get('target_mae_bpm', float('nan')):.3f}"
                )
        except Exception as exc:
            failures.append({"source": str(acc_path), "error": repr(exc)})
            print(f"  ERROR: {exc}")

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(["position", "target_rpm"], na_position="last").reset_index(drop=True)
    write_summary_report(summary, args.output_dir, cfg)
    if failures:
        (args.output_dir / "failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Completed with {len(failures)} failure(s). See {args.output_dir / 'failures.json'}")
        return 2

    print(f"Done. Summary: {args.output_dir / 'summary.csv'}")
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
