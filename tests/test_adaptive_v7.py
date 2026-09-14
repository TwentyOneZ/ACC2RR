import unittest

import numpy as np

from acc2rr.adaptive_v7 import compute_ambiguity_evidence, select_hybrid_modes


class AdaptiveV7RegressionTests(unittest.TestCase):
    def _row(
        self,
        *,
        psd=20.0,
        acf=20.0,
        ratio=1.2,
        variance=0.96,
        confidence=0.70,
        agreement=0.90,
    ):
        return {
            "rr_psd_bpm": psd,
            "rr_acf_bpm": acf,
            "reference_fundamental_to_2x_ratio": ratio,
            "pca_explained_variance_ratio": variance,
            "benchmark_confidence": confidence,
            "agreement_quality": agreement,
        }

    def test_healthy_pc1_stays_low_ambiguity_and_ineligible(self):
        pc1 = self._row()
        optimized = self._row(confidence=0.74)
        evidence = compute_ambiguity_evidence(pc1, optimized)
        self.assertLess(float(evidence["ambiguity_score"]), 0.25)
        self.assertFalse(bool(evidence["adaptive_eligible"]))
        self.assertGreater(float(evidence["classic_score"]), float(evidence["adaptive_score"]))

    def test_harmonic_low_variance_case_activates_adaptive_evidence(self):
        pc1 = self._row(
            psd=40.0,
            acf=20.0,
            ratio=0.20,
            variance=0.70,
            confidence=0.48,
            agreement=0.55,
        )
        optimized = self._row(confidence=0.68)
        evidence = compute_ambiguity_evidence(pc1, optimized)
        self.assertTrue(bool(evidence["harmonic_disagreement"]))
        self.assertTrue(bool(evidence["adaptive_eligible"]))
        self.assertGreater(float(evidence["ambiguity_score"]), 0.70)
        self.assertGreater(float(evidence["adaptive_score"]), float(evidence["classic_score"]))

    def test_target_fields_do_not_change_ambiguity_or_mode_scores(self):
        pc1_a = self._row(psd=40.0, acf=20.0, ratio=0.25, variance=0.72, confidence=0.50)
        pc1_b = dict(pc1_a)
        pc1_a["target_rpm"] = 10.0
        pc1_a["target_error_bpm"] = 30.0
        pc1_b["target_rpm"] = 20.0
        pc1_b["target_error_bpm"] = 20.0
        optimized_a = self._row(confidence=0.70)
        optimized_b = dict(optimized_a)
        optimized_a["target_abs_error_bpm"] = 0.0
        optimized_b["target_abs_error_bpm"] = 99.0

        a = compute_ambiguity_evidence(pc1_a, optimized_a)
        b = compute_ambiguity_evidence(pc1_b, optimized_b)
        for key in (
            "ambiguity_score",
            "classic_score",
            "adaptive_score",
            "confidence_gain",
            "reference_ratio_evidence",
            "variance_evidence",
        ):
            self.assertAlmostEqual(float(a[key]), float(b[key]), places=12)
        self.assertEqual(bool(a["adaptive_eligible"]), bool(b["adaptive_eligible"]))

    def test_single_moderate_ambiguity_does_not_cause_mode_flap(self):
        classic = np.full(11, 0.82)
        adaptive = np.full(11, 0.12)
        eligible = np.zeros(11, dtype=bool)
        classic[5] = 0.30
        adaptive[5] = 0.82
        eligible[5] = True
        tracked = select_hybrid_modes(classic, adaptive, eligible)["mode"]
        self.assertTrue(np.all(tracked == 0))

    def test_sustained_ambiguity_switches_to_adaptive_and_back(self):
        classic = np.full(16, 0.82)
        adaptive = np.full(16, 0.12)
        eligible = np.zeros(16, dtype=bool)
        classic[5:11] = 0.18
        adaptive[5:11] = 0.90
        eligible[5:11] = True
        tracked = select_hybrid_modes(classic, adaptive, eligible)["mode"]
        self.assertTrue(np.all(tracked[:5] == 0))
        self.assertTrue(np.all(tracked[5:11] == 1))
        self.assertTrue(np.all(tracked[11:] == 0))

    def test_ineligible_adaptive_score_is_not_enough_by_itself(self):
        classic = np.full(12, 0.60)
        adaptive = np.full(12, 0.80)
        eligible = np.zeros(12, dtype=bool)
        tracked = select_hybrid_modes(classic, adaptive, eligible)["mode"]
        self.assertTrue(np.all(tracked == 0))


if __name__ == "__main__":
    unittest.main()
