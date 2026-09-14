import unittest

import numpy as np

from acc2rr.core import (
    Config, butter_bandpass_zero_phase, estimate_gravity, hampel_vector_despike,
    pca_surrogate, project_perpendicular_to_gravity,
)
from acc2rr.estimators import estimate_surrogate, motion_quality


def synthetic_h10(rr_bpm: float, duration_s: float = 120.0, fs: float = 50.0, seed: int = 7, strong_harmonic: bool = False):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    f = rr_bpm / 60.0

    # Mean gravity roughly matches a non-axis-perfect chest strap orientation.
    g = np.array([-0.94, 0.21, 0.26])
    g /= np.linalg.norm(g)
    transverse = np.array([0.2, 0.9, -0.3])
    transverse -= transverse.dot(g) * g
    transverse /= np.linalg.norm(transverse)
    transverse2 = np.cross(g, transverse)

    fundamental = np.sin(2 * np.pi * f * t)
    harmonic_amp = 1.25 if strong_harmonic else 0.25
    waveform = fundamental + harmonic_amp * np.sin(4 * np.pi * f * t + 0.4)

    # Respiratory movement/tilt in mg plus unrelated sensor noise.
    xyz = 1000.0 * g + 18.0 * waveform[:, None] * transverse + 5.0 * np.sin(2 * np.pi * f * t + 0.8)[:, None] * transverse2
    xyz += rng.normal(0.0, 2.0, size=xyz.shape)

    # Sparse impulsive artifacts to exercise Hampel cleaning.
    spike_idx = rng.choice(len(t), size=8, replace=False)
    xyz[spike_idx] += rng.normal(0, 120, size=(len(spike_idx), 3))
    return t, xyz


class SyntheticPipelineTests(unittest.TestCase):
    def run_case(self, rr_bpm: float, strong_harmonic: bool = False):
        fs = 50.0
        cfg = Config(window_s=30.0)
        _, xyz = synthetic_h10(rr_bpm, fs=fs, strong_harmonic=strong_harmonic)
        clean, mask = hampel_vector_despike(xyz, fs, cfg.hampel_window_s, cfg.hampel_n_sigma)
        self.assertGreater(mask.sum(), 0)
        gravity, unit = estimate_gravity(clean)
        projected = project_perpendicular_to_gravity(clean, gravity, unit)
        filtered = butter_bandpass_zero_phase(projected, fs, cfg.fmin_hz, cfg.fmax_hz, cfg.filter_order)
        surrogate, eigvals, _, ratio = pca_surrogate(filtered)
        mq, _, _ = motion_quality(projected, fs, cfg)
        est, _, _ = estimate_surrogate(surrogate, eigvals, ratio, fs, cfg, movement_quality=mq)
        return est

    def test_10_rpm(self):
        est = self.run_case(10.0)
        self.assertLess(abs(est.rr_final_bpm - 10.0), 1.0)

    def test_20_rpm(self):
        est = self.run_case(20.0)
        self.assertLess(abs(est.rr_final_bpm - 20.0), 1.0)

    def test_strong_second_harmonic_still_finds_fundamental(self):
        est = self.run_case(10.0, strong_harmonic=True)
        self.assertLess(abs(est.rr_final_bpm - 10.0), 1.5)


if __name__ == "__main__":
    unittest.main()
