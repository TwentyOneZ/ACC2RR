import unittest

import numpy as np

from acc2rr.temporal_v4 import harmonic_viterbi_track_v4


def rr_candidates(
    score20: float,
    score40: float,
    acf20: float,
    acf40: float,
):
    return [
        {
            "rr_bpm": 20.0,
            "score": float(score20),
            "spectral_support": 0.0,
            "acf_support": float(acf20),
            "harmonic_support": 0.0,
            "soft_alias_bonus": 0.0,
        },
        {
            "rr_bpm": 40.0,
            "score": float(score40),
            "spectral_support": 0.0,
            "acf_support": float(acf40),
            "harmonic_support": 0.0,
            "soft_alias_bonus": 0.0,
        },
    ]


class TemporalTrackerV4RegressionTests(unittest.TestCase):
    def test_weak_global_prior_rejects_low_confidence_majority_harmonic_alias(self):
        # Mirrors the shape seen in lado/20rpm: the local 40-rpm candidate wins
        # many low-confidence windows, while ACF support remains stronger at 20.
        candidate_sets = []
        confidence = []

        candidate_sets += [rr_candidates(0.55, 0.72, 0.50, 0.28) for _ in range(7)]
        confidence += [0.48] * 7

        candidate_sets += [rr_candidates(0.72, 0.55, 0.55, 0.25) for _ in range(4)]
        confidence += [0.68] * 4

        candidate_sets += [rr_candidates(0.55, 0.72, 0.50, 0.28) for _ in range(45)]
        confidence += [0.48] * 45

        candidate_sets += [rr_candidates(0.78, 0.45, 0.60, 0.20) for _ in range(20)]
        confidence += [0.70] * 20

        tracked = harmonic_viterbi_track_v4(
            candidate_sets,
            np.asarray(confidence, dtype=float),
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.64,
            hop_s=1.0,
        )["rr_bpm"]

        self.assertGreater(np.mean(np.abs(tracked - 20.0) < 0.5), 0.95)

    def test_sustained_true_20_to_40_change_overrides_prior(self):
        candidate_sets = (
            [rr_candidates(0.85, 0.30, 0.80, 0.20) for _ in range(20)]
            + [rr_candidates(0.25, 0.92, 0.20, 0.90) for _ in range(60)]
        )
        confidence = np.asarray([0.90] * len(candidate_sets), dtype=float)

        tracked = harmonic_viterbi_track_v4(
            candidate_sets,
            confidence,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.80,
            hop_s=1.0,
        )["rr_bpm"]

        self.assertGreater(np.mean(np.abs(tracked[:20] - 20.0) < 0.5), 0.95)
        self.assertGreater(np.mean(np.abs(tracked[-40:] - 40.0) < 0.5), 0.95)

    def test_wrong_global_prior_is_not_a_lock(self):
        # A deliberately wrong full-record prior must be overcome when every
        # window strongly and confidently supports a different true rate.
        candidate_sets = [
            rr_candidates(0.20, 0.95, 0.15, 0.90) for _ in range(50)
        ]
        confidence = np.asarray([0.95] * len(candidate_sets), dtype=float)

        tracked = harmonic_viterbi_track_v4(
            candidate_sets,
            confidence,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.85,
            hop_s=1.0,
        )["rr_bpm"]

        self.assertGreater(np.mean(np.abs(tracked - 40.0) < 0.5), 0.95)

    def test_low_confidence_emissions_have_less_influence(self):
        # Same local score advantage for 40 rpm, but only the high-confidence
        # block should be capable of overcoming continuity/prior evidence.
        low_conf_sets = [
            rr_candidates(0.58, 0.74, 0.52, 0.25) for _ in range(25)
        ]
        low_conf = np.asarray([0.35] * len(low_conf_sets), dtype=float)
        low_track = harmonic_viterbi_track_v4(
            low_conf_sets,
            low_conf,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.75,
            hop_s=1.0,
        )["rr_bpm"]
        self.assertGreater(np.mean(np.abs(low_track - 20.0) < 0.5), 0.95)

        high_conf_sets = [
            rr_candidates(0.20, 0.92, 0.15, 0.88) for _ in range(50)
        ]
        high_conf = np.asarray([0.95] * len(high_conf_sets), dtype=float)
        high_track = harmonic_viterbi_track_v4(
            high_conf_sets,
            high_conf,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.75,
            hop_s=1.0,
        )["rr_bpm"]
        self.assertGreater(np.mean(np.abs(high_track - 40.0) < 0.5), 0.95)


if __name__ == "__main__":
    unittest.main()
