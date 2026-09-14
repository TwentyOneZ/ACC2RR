import unittest

import numpy as np

from acc2rr.core import Config
from acc2rr.surrogate_v5 import (
    PC12_ANGLE_STEP_DEG,
    benchmark_window,
    evaluate_surrogate,
)


class SurrogateBenchmarkV5Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.fs = 50.0
        self.t = np.arange(0.0, 30.0, 1.0 / self.fs)

    def test_reference_rr_does_not_change_estimate_or_selection_score(self):
        x = (
            np.sin(2.0 * np.pi * (20.0 / 60.0) * self.t)
            + 0.30 * np.sin(2.0 * np.pi * (40.0 / 60.0) * self.t + 0.3)
        )
        a = evaluate_surrogate(x, self.fs, self.cfg, reference_rr_bpm=10.0)
        b = evaluate_surrogate(x, self.fs, self.cfg, reference_rr_bpm=20.0)

        self.assertAlmostEqual(a["rr_consensus_bpm"], b["rr_consensus_bpm"], places=10)
        self.assertAlmostEqual(a["benchmark_confidence"], b["benchmark_confidence"], places=10)
        self.assertAlmostEqual(a["spectral_quality"], b["spectral_quality"], places=10)
        self.assertAlmostEqual(a["acf_quality"], b["acf_quality"], places=10)

    def test_benchmark_emits_expected_surrogates_and_complete_angle_scan(self):
        f = 20.0 / 60.0
        filtered = np.column_stack(
            [
                4.0 * np.sin(2.0 * np.pi * f * self.t),
                2.0 * np.sin(2.0 * np.pi * f * self.t + 0.7),
                0.5 * np.sin(2.0 * np.pi * 2.0 * f * self.t + 0.2),
            ]
        )
        records, angles = benchmark_window(
            filtered,
            self.fs,
            self.cfg,
            reference_rr_bpm=20.0,
        )

        names = {record["surrogate"] for record in records}
        expected = {
            "pc1",
            "pc2",
            "pc3",
            "axis_x",
            "axis_y",
            "axis_z",
            "vector_magnitude",
            "best_axis",
            "best_pca_component",
            "optimized_pc12",
        }
        self.assertTrue(expected.issubset(names))
        self.assertEqual(len(angles), len(range(0, 180, PC12_ANGLE_STEP_DEG)))

    def test_optimized_pc12_is_the_highest_scoring_scanned_direction(self):
        rng = np.random.default_rng(4)
        f = 20.0 / 60.0
        filtered = np.column_stack(
            [
                5.0 * np.sin(2.0 * np.pi * 2.0 * f * self.t)
                + rng.normal(0.0, 1.2, len(self.t)),
                3.0 * np.sin(2.0 * np.pi * f * self.t)
                + rng.normal(0.0, 0.25, len(self.t)),
                rng.normal(0.0, 0.08, len(self.t)),
            ]
        )
        records, angles = benchmark_window(
            filtered,
            self.fs,
            self.cfg,
            reference_rr_bpm=20.0,
        )
        optimized = next(r for r in records if r["surrogate"] == "optimized_pc12")
        max_score = max(float(r["benchmark_confidence"]) for r in angles)
        self.assertAlmostEqual(float(optimized["benchmark_confidence"]), max_score, places=12)

    def test_reference_harmonic_ratio_reports_dominant_second_harmonic(self):
        f = 20.0 / 60.0
        x = (
            np.sin(2.0 * np.pi * f * self.t)
            + 2.0 * np.sin(2.0 * np.pi * 2.0 * f * self.t + 0.2)
        )
        result = evaluate_surrogate(x, self.fs, self.cfg, reference_rr_bpm=20.0)
        ratio = float(result["reference_fundamental_to_2x_ratio"])
        self.assertGreater(ratio, 0.10)
        self.assertLess(ratio, 0.50)


if __name__ == "__main__":
    unittest.main()
