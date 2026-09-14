import unittest

import numpy as np

from acc2rr.core import Config
from acc2rr.temporal_v6 import (
    build_v6_states_for_window,
    joint_viterbi_track_v6,
    orientation_distance_rad,
)


def state(
    rr_bpm: float,
    confidence: float,
    direction: tuple[float, float, float],
    *,
    acf_quality: float = 0.75,
    angle_deg: float = 0.0,
):
    direction_array = np.asarray(direction, dtype=float)
    direction_array /= np.linalg.norm(direction_array)
    return {
        "angle_deg": float(angle_deg),
        "rr_bpm": float(rr_bpm),
        "score": float(confidence),
        "benchmark_confidence": float(confidence),
        "spectral_quality": float(confidence),
        "acf_quality": float(acf_quality),
        "agreement_quality": float(confidence),
        "consensus_score": float(confidence),
        "reference_fundamental_to_2x_ratio": 1.0,
        "pca_pc1_variance_ratio": 0.7,
        "pca_pc2_variance_ratio": 0.3,
        "direction_x": float(direction_array[0]),
        "direction_y": float(direction_array[1]),
        "direction_z": float(direction_array[2]),
    }


class TemporalTrackerV6RegressionTests(unittest.TestCase):
    def test_orientation_distance_is_sign_invariant(self):
        a = np.array([1.0, 2.0, -0.5])
        b = -a
        self.assertAlmostEqual(orientation_distance_rad(a, b), 0.0, places=12)

    def test_window_state_scan_emits_all_angles_with_unit_physical_directions(self):
        cfg = Config()
        fs = 50.0
        t = np.arange(0.0, 30.0, 1.0 / fs)
        f = 20.0 / 60.0
        filtered = np.column_stack(
            [
                5.0 * np.sin(2.0 * np.pi * f * t),
                3.0 * np.sin(2.0 * np.pi * f * t + 0.7),
                0.4 * np.sin(2.0 * np.pi * 2.0 * f * t + 0.2),
            ]
        )
        states = build_v6_states_for_window(
            filtered,
            fs,
            cfg,
            reference_rr_bpm=20.0,
        )
        self.assertEqual(len(states), 36)
        self.assertEqual(
            [int(s["angle_deg"]) for s in states],
            list(range(0, 180, 5)),
        )
        for candidate in states:
            vector = np.array(
                [candidate["direction_x"], candidate["direction_y"], candidate["direction_z"]]
            )
            self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=10)

    def test_joint_tracker_rejects_small_angular_jitter(self):
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        state_sets = []
        for index in range(24):
            if index % 2 == 0:
                state_sets.append(
                    [
                        state(20.0, 0.72, x_axis, angle_deg=0.0),
                        state(20.0, 0.76, y_axis, angle_deg=90.0),
                    ]
                )
            else:
                state_sets.append(
                    [
                        state(20.0, 0.76, x_axis, angle_deg=0.0),
                        state(20.0, 0.72, y_axis, angle_deg=90.0),
                    ]
                )

        tracked = joint_viterbi_track_v6(
            state_sets,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.8,
            hop_s=1.0,
        )
        physical_steps = np.degrees(tracked["angular_distance_rad"])
        self.assertLess(float(np.nanmax(physical_steps)), 1.0)
        self.assertTrue(np.all(np.abs(tracked["rr_bpm"] - 20.0) < 0.5))

    def test_joint_tracker_allows_sustained_orientation_change(self):
        x_axis = (1.0, 0.0, 0.0)
        y_axis = (0.0, 1.0, 0.0)
        state_sets = []
        for _ in range(12):
            state_sets.append(
                [
                    state(20.0, 0.90, x_axis, angle_deg=0.0),
                    state(20.0, 0.30, y_axis, angle_deg=90.0),
                ]
            )
        for _ in range(45):
            state_sets.append(
                [
                    state(20.0, 0.30, x_axis, angle_deg=0.0),
                    state(20.0, 0.90, y_axis, angle_deg=90.0),
                ]
            )

        tracked = joint_viterbi_track_v6(
            state_sets,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.8,
            hop_s=1.0,
        )
        self.assertLess(abs(float(tracked["angle_deg"][0]) - 0.0), 1.0)
        self.assertGreater(
            np.mean(np.abs(tracked["angle_deg"][-30:] - 90.0) < 1.0),
            0.95,
        )

    def test_joint_tracker_allows_sustained_true_harmonic_rr_change(self):
        direction = (1.0, 0.0, 0.0)
        state_sets = []
        for _ in range(20):
            state_sets.append(
                [
                    state(20.0, 0.90, direction, acf_quality=0.85),
                    state(40.0, 0.30, direction, acf_quality=0.40),
                ]
            )
        for _ in range(60):
            state_sets.append(
                [
                    state(20.0, 0.30, direction, acf_quality=0.40),
                    state(40.0, 0.92, direction, acf_quality=0.90),
                ]
            )

        tracked = joint_viterbi_track_v6(
            state_sets,
            global_prior_rr_bpm=20.0,
            global_prior_confidence=0.8,
            hop_s=1.0,
        )
        self.assertTrue(np.all(np.abs(tracked["rr_bpm"][:20] - 20.0) < 0.5))
        self.assertGreater(
            np.mean(np.abs(tracked["rr_bpm"][-40:] - 40.0) < 0.5),
            0.95,
        )


if __name__ == "__main__":
    unittest.main()
