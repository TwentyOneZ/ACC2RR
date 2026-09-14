import unittest

import numpy as np

from acc2rr.adaptive_v8 import (
    V8_HARD_OFF_WINDOWS,
    V8_HARD_ON_WINDOWS,
    ambiguity_to_desired_alpha,
    fuse_observations,
    hard_gate_schmitt,
    smooth_fusion_alpha,
)


class AdaptiveV8RegressionTests(unittest.TestCase):
    def test_single_high_ambiguity_window_does_not_activate_hard_gate(self):
        ambiguity = np.array([0.05, 0.10, 0.90, 0.10, 0.05])
        tracked = hard_gate_schmitt(ambiguity)
        self.assertFalse(bool(np.any(tracked["adaptive"])))
        self.assertFalse(bool(np.any(tracked["switch"])))

    def test_hard_gate_uses_asymmetric_sustained_on_and_off_hysteresis(self):
        high = np.full(V8_HARD_ON_WINDOWS, 0.80)
        middle = np.full(4, 0.38)
        almost_low = np.full(V8_HARD_OFF_WINDOWS - 1, 0.10)
        final_low = np.array([0.10])
        ambiguity = np.concatenate([np.array([0.05]), high, middle, almost_low, final_low])
        tracked = hard_gate_schmitt(ambiguity)
        mode = tracked["adaptive"]

        # Starts classic, activates only after the required consecutive highs.
        self.assertFalse(bool(mode[0]))
        self.assertFalse(bool(mode[1]))
        self.assertTrue(bool(mode[V8_HARD_ON_WINDOWS]))

        # Intermediate ambiguity does not prematurely release adaptation.
        middle_start = 1 + V8_HARD_ON_WINDOWS
        middle_stop = middle_start + len(middle)
        self.assertTrue(bool(np.all(mode[middle_start:middle_stop])))

        # Fewer than the required low windows are insufficient to switch off.
        low_start = middle_stop
        low_stop = low_start + len(almost_low)
        self.assertTrue(bool(np.all(mode[low_start:low_stop])))
        self.assertFalse(bool(mode[-1]))
        self.assertEqual(int(np.sum(tracked["switch"])), 2)

    def test_hard_gate_ignores_mid_band_ambiguity_while_classic(self):
        ambiguity = np.full(20, 0.35)
        tracked = hard_gate_schmitt(ambiguity)
        self.assertTrue(np.all(tracked["adaptive"] == 0))

    def test_soft_alpha_mapping_is_bounded_monotonic_and_saturates(self):
        ambiguity = np.linspace(0.0, 1.0, 101)
        alpha = ambiguity_to_desired_alpha(ambiguity)
        self.assertAlmostEqual(float(alpha[0]), 0.0, places=12)
        self.assertAlmostEqual(float(alpha[-1]), 1.0, places=12)
        self.assertTrue(np.all(np.diff(alpha) >= -1e-12))
        self.assertTrue(np.all((alpha >= 0.0) & (alpha <= 1.0)))

    def test_soft_alpha_engages_faster_than_it_releases(self):
        desired = np.concatenate([np.zeros(3), np.ones(3), np.zeros(3)])
        alpha = smooth_fusion_alpha(desired)
        rise_first_step = alpha[3] - alpha[2]
        fall_first_step = alpha[6] - alpha[5]
        self.assertGreater(rise_first_step, 0.0)
        self.assertLess(fall_first_step, 0.0)
        self.assertGreater(rise_first_step, abs(fall_first_step))
        self.assertGreater(alpha[-1], 0.0)  # release is deliberately gradual

    def test_fusion_endpoints_reproduce_classic_and_adaptive_exactly(self):
        classic_rr = np.array([10.0, 20.0, 30.0])
        adaptive_rr = np.array([11.0, 21.0, 31.0])
        classic_conf = np.array([0.9, 0.8, 0.7])
        adaptive_conf = np.array([0.5, 0.6, 0.7])

        rr_classic, conf_classic = fuse_observations(
            classic_rr, adaptive_rr, classic_conf, adaptive_conf, np.zeros(3)
        )
        rr_adaptive, conf_adaptive = fuse_observations(
            classic_rr, adaptive_rr, classic_conf, adaptive_conf, np.ones(3)
        )

        np.testing.assert_allclose(rr_classic, classic_rr)
        np.testing.assert_allclose(conf_classic, classic_conf)
        np.testing.assert_allclose(rr_adaptive, adaptive_rr)
        np.testing.assert_allclose(conf_adaptive, adaptive_conf)

    def test_soft_fusion_remains_between_both_observations(self):
        classic_rr = np.array([40.0, 10.0, 22.0])
        adaptive_rr = np.array([20.0, 12.0, 18.0])
        classic_conf = np.full(3, 0.6)
        adaptive_conf = np.full(3, 0.8)
        alpha = np.array([0.25, 0.50, 0.75])
        rr, confidence = fuse_observations(
            classic_rr, adaptive_rr, classic_conf, adaptive_conf, alpha
        )
        lower = np.minimum(classic_rr, adaptive_rr)
        upper = np.maximum(classic_rr, adaptive_rr)
        self.assertTrue(np.all(rr >= lower))
        self.assertTrue(np.all(rr <= upper))
        self.assertTrue(np.all(confidence >= 0.0))
        self.assertTrue(np.all(confidence <= 1.0))


if __name__ == "__main__":
    unittest.main()
