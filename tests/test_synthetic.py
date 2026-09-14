import unittest

import numpy as np

from acc2rr.core import (
    Config, butter_bandpass_zero_phase, estimate_gravity, hampel_vector_despike,
    pca_surrogate, project_perpendicular_to_gravity,
)
from acc2rr.estimators import consensus_estimate, estimate_surrogate, motion_quality


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


def fake_psd(cfg: Config, peaks: list[tuple[float, float]], selected_hz: float):
    """Build a deterministic PSD fixture with Gaussian peaks.

    peaks contains (frequency_hz, normalized_amplitude). This lets the tests
    reproduce the observed real-data failure mode without relying on the exact
    biomechanical waveform that generated it.
    """
    freq = np.linspace(cfg.fmin_hz, cfg.fmax_hz, 4000)
    power = np.full_like(freq, 1e-5)
    for f_hz, amplitude in peaks:
        power += amplitude * np.exp(-0.5 * ((freq - f_hz) / 0.006) ** 2)
    p_max = float(np.max(power))
    return {
        "freq_hz": freq,
        "power": power,
        "p_max": p_max,
        "candidate_freqs_hz": [f for f, _ in sorted(peaks, key=lambda item: item[1], reverse=True)],
        "f_selected_hz": selected_hz,
        "rr_bpm": 60.0 * selected_hz,
    }


def fake_acf(rr_bpm: float, quality: float, points: list[tuple[float, float]]):
    """Build an ACF fixture from (lag_seconds, amplitude) peaks."""
    lags = np.linspace(0.0, 12.5, 5001)
    acf = np.zeros_like(lags)
    for lag_s, amplitude in points:
        acf += amplitude * np.exp(-0.5 * ((lags - lag_s) / 0.06) ** 2)
    acf[0] = 1.0
    return {
        "lags_s": lags,
        "acf": acf,
        "rr_bpm": rr_bpm,
        "quality": quality,
        "candidate_freqs_hz": [rr_bpm / 60.0],
        "selected_lag_s": 60.0 / rr_bpm,
    }


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


class HarmonicConsensusRegressionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_consensus_resolves_realistic_second_harmonic_alias(self):
        # Mirrors lado/20rpm: PSD at ~40 rpm is dominant, but the fundamental
        # retains ~12% of peak power and ACF strongly supports 20 rpm.
        psd = fake_psd(
            self.cfg,
            [(40.0 / 60.0, 1.0), (20.0 / 60.0, 0.121)],
            selected_hz=40.0 / 60.0,
        )
        acf = fake_acf(
            rr_bpm=20.0,
            quality=0.65,
            points=[(3.0, 0.50), (1.5, 0.30)],
        )
        rr, score, agreement = consensus_estimate(psd, acf, self.cfg)
        self.assertLess(abs(rr - 20.0), 0.5)
        self.assertGreater(score, 0.5)
        self.assertGreaterEqual(agreement, 0.70)

    def test_consensus_does_not_halve_true_fundamental(self):
        psd = fake_psd(
            self.cfg,
            [(20.0 / 60.0, 1.0), (10.0 / 60.0, 0.04)],
            selected_hz=20.0 / 60.0,
        )
        acf = fake_acf(
            rr_bpm=20.0,
            quality=0.80,
            points=[(3.0, 0.80)],
        )
        rr, _, agreement = consensus_estimate(psd, acf, self.cfg)
        self.assertLess(abs(rr - 20.0), 0.5)
        self.assertGreater(agreement, 0.95)

    def test_consensus_ignores_weak_misleading_acf_subharmonic(self):
        # PSD says 20 rpm while a deliberately weak/noisy ACF says 10 rpm.
        # The harmonic gate must remain closed, preventing false halving.
        psd = fake_psd(
            self.cfg,
            [(20.0 / 60.0, 1.0), (10.0 / 60.0, 0.12)],
            selected_hz=20.0 / 60.0,
        )
        acf = fake_acf(
            rr_bpm=10.0,
            quality=0.20,
            points=[(6.0, 0.20), (3.0, 0.60)],
        )
        rr, _, _ = consensus_estimate(psd, acf, self.cfg)
        self.assertLess(abs(rr - 20.0), 0.5)

    def test_consensus_can_resolve_third_harmonic_alias(self):
        psd = fake_psd(
            self.cfg,
            [(36.0 / 60.0, 1.0), (12.0 / 60.0, 0.10)],
            selected_hz=36.0 / 60.0,
        )
        acf = fake_acf(
            rr_bpm=12.0,
            quality=0.70,
            points=[(5.0, 0.55), (60.0 / 36.0, 0.25)],
        )
        rr, score, agreement = consensus_estimate(psd, acf, self.cfg)
        self.assertLess(abs(rr - 12.0), 0.5)
        self.assertGreater(score, 0.5)
        self.assertGreaterEqual(agreement, 0.60)


if __name__ == "__main__":
    unittest.main()
