from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal
from scipy.fft import next_fast_len

from .core import Config, Estimate, EPS


def _parabolic_peak_frequency(freq: np.ndarray, power: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(power) - 1:
        return float(freq[idx])
    y = np.log(np.maximum(power[idx - 1 : idx + 2], EPS))
    denom = y[0] - 2.0 * y[1] + y[2]
    if abs(denom) < EPS:
        return float(freq[idx])
    delta = 0.5 * (y[0] - y[2]) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(freq[idx] + delta * (freq[idx + 1] - freq[idx]))


def psd_estimate(
    surrogate: np.ndarray,
    fs_hz: float,
    cfg: Config,
) -> dict[str, Any]:
    n = len(surrogate)
    nperseg = min(n, max(64, int(round(cfg.welch_segment_s * fs_hz))))
    noverlap = nperseg // 2 if n > nperseg else 0
    nfft = next_fast_len(max(nperseg * 4, int(math.ceil(fs_hz / 0.005))))
    freq, power = signal.welch(
        surrogate,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend="constant",
        scaling="density",
    )
    band = (freq >= cfg.fmin_hz) & (freq <= cfg.fmax_hz)
    if band.sum() < 3:
        raise ValueError("Respiratory frequency band contains too few PSD bins.")

    f_band = freq[band]
    p_band = power[band]
    idx_local = int(np.argmax(p_band))
    idx_global = int(np.flatnonzero(band)[idx_local])
    f_raw = _parabolic_peak_frequency(freq, power, idx_global)

    p_max = float(np.max(p_band))
    p_interp_raw = float(np.interp(f_raw, freq, power))
    f_selected = f_raw
    harmonic_corrected = False

    # Purely spectral fundamental check. The independent ACF estimate remains
    # untouched; final consensus can therefore diagnose disagreement.
    half_f = 0.5 * f_raw
    half_ratio = 0.0
    if half_f >= cfg.fmin_hz:
        p_half = float(np.interp(half_f, freq, power))
        half_ratio = p_half / max(p_interp_raw, EPS)
        if half_ratio >= cfg.harmonic_half_power_ratio:
            f_selected = half_f
            harmonic_corrected = True

    # Spectral concentration + normalized prominence form an interpretable
    # quality score in [0, 1].
    local = np.abs(f_band - f_selected) <= cfg.local_peak_halfwidth_hz
    band_power = float(np.trapezoid(p_band, f_band))
    local_power = float(np.trapezoid(p_band[local], f_band[local])) if local.sum() >= 2 else 0.0
    concentration = local_power / max(band_power, EPS)

    peaks, props = signal.find_peaks(p_band, prominence=max(p_max * 0.01, EPS))
    if len(peaks):
        nearest = int(np.argmin(np.abs(f_band[peaks] - f_selected)))
        prominence = float(props["prominences"][nearest]) / max(p_max, EPS)
    else:
        prominence = 0.0

    prob = p_band / max(float(np.sum(p_band)), EPS)
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, EPS)))) / max(math.log(len(prob)), EPS)
    inverse_entropy = 1.0 - entropy
    spectral_quality = float(
        np.clip(0.45 * concentration + 0.35 * prominence + 0.20 * inverse_entropy, 0.0, 1.0)
    )

    # Candidate peaks used by the harmonic-aware consensus stage.
    if len(peaks):
        order = np.argsort(p_band[peaks])[::-1][:8]
        peak_freqs = [float(f_band[peaks[i]]) for i in order]
    else:
        peak_freqs = [float(f_raw)]

    return {
        "freq_hz": freq,
        "power": power,
        "band_mask": band,
        "f_raw_hz": f_raw,
        "f_selected_hz": f_selected,
        "rr_raw_bpm": 60.0 * f_raw,
        "rr_bpm": 60.0 * f_selected,
        "harmonic_corrected": harmonic_corrected,
        "half_power_ratio": half_ratio,
        "quality": spectral_quality,
        "concentration": concentration,
        "prominence": prominence,
        "inverse_entropy": inverse_entropy,
        "candidate_freqs_hz": peak_freqs,
        "p_max": p_max,
    }


def autocorrelation_estimate(
    surrogate: np.ndarray,
    fs_hz: float,
    cfg: Config,
) -> dict[str, Any]:
    x = np.asarray(surrogate, float)
    x = x - np.mean(x)
    std = float(np.std(x))
    if std < EPS:
        return {
            "lags_s": np.array([0.0]),
            "acf": np.array([1.0]),
            "rr_bpm": float("nan"),
            "quality": 0.0,
            "candidate_freqs_hz": [],
            "selected_lag_s": float("nan"),
        }
    x /= std
    n = len(x)
    corr = signal.fftconvolve(x, x[::-1], mode="full")[n - 1 :]
    overlap = np.arange(n, 0, -1, dtype=float)
    corr = corr / overlap
    corr = corr / max(corr[0], EPS)
    lags = np.arange(n, dtype=float) / fs_hz

    min_lag = 1.0 / cfg.fmax_hz
    max_lag = min(1.0 / cfg.fmin_hz, lags[-1])
    mask = (lags >= min_lag) & (lags <= max_lag)
    idx_band = np.flatnonzero(mask)
    if len(idx_band) < 3:
        return {
            "lags_s": lags,
            "acf": corr,
            "rr_bpm": float("nan"),
            "quality": 0.0,
            "candidate_freqs_hz": [],
            "selected_lag_s": float("nan"),
        }

    acf_band = corr[idx_band]
    min_peak_distance = max(1, int(round(0.35 * fs_hz)))
    peaks, props = signal.find_peaks(acf_band, prominence=0.02, distance=min_peak_distance)
    if not len(peaks):
        local = int(np.argmax(acf_band))
        selected_global = int(idx_band[local])
        selected_height = float(corr[selected_global])
        selected_prom = 0.0
        candidate_global = np.array([selected_global])
    else:
        heights = acf_band[peaks]
        prominences = props["prominences"]
        max_h = max(float(np.max(heights)), EPS)
        max_p = max(float(np.max(prominences)), EPS)
        strong = (heights >= 0.55 * max_h) & (prominences >= 0.25 * max_p)
        strong_peaks = peaks[strong]
        if len(strong_peaks):
            # Earliest strong periodic recurrence is usually the fundamental.
            selected_local_peak = int(strong_peaks[np.argmin(strong_peaks)])
        else:
            score = np.clip(heights, 0, None) * np.maximum(prominences, EPS)
            selected_local_peak = int(peaks[int(np.argmax(score))])
        selected_global = int(idx_band[selected_local_peak])
        selected_height = float(corr[selected_global])
        match = int(np.where(peaks == selected_local_peak)[0][0])
        selected_prom = float(prominences[match])
        candidate_global = idx_band[peaks]

    selected_lag = float(lags[selected_global])
    rr_bpm = 60.0 / selected_lag if selected_lag > 0 else float("nan")
    acf_quality = float(
        np.clip(0.70 * max(selected_height, 0.0) + 0.30 * min(selected_prom, 1.0), 0.0, 1.0)
    )
    candidate_freqs = [float(1.0 / lags[i]) for i in candidate_global if lags[i] > 0]

    return {
        "lags_s": lags,
        "acf": corr,
        "rr_bpm": rr_bpm,
        "quality": acf_quality,
        "candidate_freqs_hz": candidate_freqs,
        "selected_lag_s": selected_lag,
        "selected_height": selected_height,
        "selected_prominence": selected_prom,
    }


def _normalized_psd_at(f_hz: float, psd: dict[str, Any]) -> float:
    if not np.isfinite(f_hz):
        return 0.0
    p = float(np.interp(f_hz, psd["freq_hz"], psd["power"], left=0.0, right=0.0))
    return float(np.clip(p / max(psd["p_max"], EPS), 0.0, 1.0))


def _acf_at_frequency(f_hz: float, acf: dict[str, Any]) -> float:
    if not np.isfinite(f_hz) or f_hz <= 0:
        return 0.0
    lag = 1.0 / f_hz
    value = float(np.interp(lag, acf["lags_s"], acf["acf"], left=0.0, right=0.0))
    return float(np.clip(value, 0.0, 1.0))


def _harmonic_alias_context(
    psd: dict[str, Any],
    acf: dict[str, Any],
    cfg: Config,
) -> dict[str, float | int | bool]:
    """Detect a trustworthy PSD-vs-ACF integer-harmonic disagreement.

    This is deliberately conservative. It does not simply lower the global PSD
    half-power threshold. Instead it requires three independent observations:

    1) the selected PSD frequency is approximately 2x or 3x the ACF frequency;
    2) the ACF has credible quality/support at the lower frequency;
    3) the PSD still contains non-trivial direct energy at that lower frequency.

    This lets a strong harmonic support the fundamental without allowing a weak,
    noisy ACF subharmonic to halve an otherwise valid respiratory estimate.
    """
    rr_psd = float(psd.get("rr_bpm", float("nan")))
    rr_acf = float(acf.get("rr_bpm", float("nan")))
    if not (np.isfinite(rr_psd) and np.isfinite(rr_acf) and rr_psd > 0 and rr_acf > 0):
        return {"active": False, "multiple": 1, "acf_f_hz": float("nan")}

    ratio = rr_psd / rr_acf
    best_multiple = min((2, 3), key=lambda m: abs(ratio - m))
    relative_error = abs(ratio - best_multiple) / best_multiple
    if relative_error > 0.08:
        return {
            "active": False,
            "multiple": int(best_multiple),
            "acf_f_hz": rr_acf / 60.0,
            "ratio": ratio,
            "relative_error": relative_error,
        }

    acf_f = rr_acf / 60.0
    acf_support = _acf_at_frequency(acf_f, acf)
    acf_quality = float(acf.get("quality", 0.0))
    direct_spectral = _normalized_psd_at(acf_f, psd)
    harmonic_spectral = _normalized_psd_at(best_multiple * acf_f, psd)

    # For a second-harmonic alias, require at least half of the original
    # spectral-only half-power threshold (0.10 with the default config). For a
    # third harmonic, 0.06 is sufficient because third-order energy can dominate
    # a visibly present fundamental more strongly.
    if best_multiple == 2:
        min_direct_spectral = max(0.08, 0.50 * cfg.harmonic_half_power_ratio)
    else:
        min_direct_spectral = max(0.05, 0.30 * cfg.harmonic_half_power_ratio)

    active = bool(
        acf_quality >= 0.50
        and acf_support >= 0.40
        and direct_spectral >= min_direct_spectral
        and harmonic_spectral >= 0.50
    )
    return {
        "active": active,
        "multiple": int(best_multiple),
        "acf_f_hz": acf_f,
        "ratio": ratio,
        "relative_error": relative_error,
        "acf_support": acf_support,
        "acf_quality": acf_quality,
        "direct_spectral": direct_spectral,
        "harmonic_spectral": harmonic_spectral,
        "min_direct_spectral": min_direct_spectral,
    }


def consensus_estimate(
    psd: dict[str, Any],
    acf: dict[str, Any],
    cfg: Config,
) -> tuple[float, float, float]:
    candidates: list[float] = []
    candidates.extend(psd.get("candidate_freqs_hz", []))
    candidates.extend(acf.get("candidate_freqs_hz", []))
    candidates.extend([psd.get("f_selected_hz", float("nan"))])
    rr_acf = acf.get("rr_bpm", float("nan"))
    if np.isfinite(rr_acf):
        candidates.append(rr_acf / 60.0)

    # Explicitly consider integer harmonic relatives. This is wider than the
    # previous f/2-only expansion, but the scoring/alias gate below remains
    # conservative and requires direct evidence at the proposed fundamental.
    expanded: list[float] = []
    for f in candidates:
        if not np.isfinite(f) or f <= 0:
            continue
        for multiplier in (1.0 / 3.0, 0.5, 1.0, 2.0, 3.0):
            candidate = multiplier * float(f)
            if cfg.fmin_hz <= candidate <= cfg.fmax_hz:
                expanded.append(candidate)

    if not expanded:
        return float("nan"), 0.0, 0.0

    # Merge near-identical candidates to avoid arbitrary scoring duplicates.
    expanded = sorted(expanded)
    merged: list[float] = []
    for f in expanded:
        if not merged or abs(f - merged[-1]) > 0.005:
            merged.append(f)
        else:
            merged[-1] = 0.5 * (merged[-1] + f)

    alias = _harmonic_alias_context(psd, acf, cfg)
    alias_active = bool(alias.get("active", False))
    alias_f = float(alias.get("acf_f_hz", float("nan")))
    alias_multiple = int(alias.get("multiple", 1))
    psd_selected_f = float(psd.get("rr_bpm", float("nan"))) / 60.0
    alias_tol_hz = max(0.008, 0.04 * alias_f) if np.isfinite(alias_f) else 0.008

    best_f = merged[0]
    best_score = -np.inf
    for f in merged:
        spectral = _normalized_psd_at(f, psd)
        acf_support = _acf_at_frequency(f, acf)

        second_harmonic = _normalized_psd_at(2.0 * f, psd) if 2.0 * f <= cfg.fmax_hz else 0.0
        third_harmonic = _normalized_psd_at(3.0 * f, psd) if 3.0 * f <= cfg.fmax_hz else 0.0
        # A strong harmonic is supporting evidence for a candidate fundamental,
        # not a replacement for direct evidence at the fundamental itself.
        harmonic = max(second_harmonic, 0.75 * third_harmonic)

        score = 0.48 * spectral + 0.42 * acf_support + 0.10 * harmonic
        if spectral < 0.04 and acf_support < 0.25:
            score *= 0.5

        if alias_active:
            # The real-data failure mode is PSD ~= 2*ACF with a strong ACF at
            # the lower frequency. Give the lower candidate a bounded, explicit
            # bonus only after the conservative alias gate has been satisfied.
            if abs(f - alias_f) <= alias_tol_hz:
                alias_bonus = 0.36 if alias_multiple == 2 else 0.31
                score += alias_bonus

            # Conversely, reduce the score of the dominant harmonic slightly.
            # This is intentionally smaller than the fundamental bonus so the
            # lower candidate must still carry direct PSD + ACF evidence.
            if np.isfinite(psd_selected_f) and abs(f - psd_selected_f) <= max(0.008, 0.04 * psd_selected_f):
                score -= 0.12 if alias_multiple == 2 else 0.10

        if score > best_score:
            best_score = score
            best_f = f

    rr_final = 60.0 * best_f
    rr_psd = float(psd.get("rr_bpm", float("nan")))
    rr_acf = float(acf.get("rr_bpm", float("nan")))

    if np.isfinite(rr_psd) and np.isfinite(rr_acf):
        diff = abs(rr_psd - rr_acf)
        agreement = float(
            math.exp(-0.5 * (diff / max(cfg.consensus_agreement_bpm, EPS)) ** 2)
        )

        # A validated integer-harmonic disagreement is not arbitrary estimator
        # disagreement. Report partial agreement after mapping the PSD harmonic
        # back to the ACF-supported fundamental. The factor below deliberately
        # keeps confidence lower than direct 1:1 agreement.
        if alias_active and alias_multiple in (2, 3):
            mapped_diff = abs(rr_psd / alias_multiple - rr_acf)
            mapped_agreement = math.exp(
                -0.5 * (mapped_diff / max(cfg.consensus_agreement_bpm, EPS)) ** 2
            )
            harmonic_ceiling = 0.75 if alias_multiple == 2 else 0.65
            agreement = max(agreement, harmonic_ceiling * mapped_agreement)
    else:
        agreement = 0.0

    return rr_final, float(np.clip(best_score, 0.0, 1.0)), float(np.clip(agreement, 0.0, 1.0))


def motion_quality(
    projected_xyz: np.ndarray,
    fs_hz: float,
    cfg: Config,
) -> tuple[float, float, float]:
    nperseg = min(len(projected_xyz), max(64, int(round(20.0 * fs_hz))))
    freq, pxx = signal.welch(projected_xyz, fs=fs_hz, axis=0, nperseg=nperseg)
    pmean = np.mean(pxx, axis=1)
    resp = (freq >= cfg.fmin_hz) & (freq <= cfg.fmax_hz)
    motion = (freq > cfg.fmax_hz) & (freq <= min(5.0, 0.5 * fs_hz * 0.95))
    resp_power = float(np.trapezoid(pmean[resp], freq[resp])) if resp.sum() >= 2 else 0.0
    motion_power = float(np.trapezoid(pmean[motion], freq[motion])) if motion.sum() >= 2 else 0.0
    snr_db = 10.0 * math.log10((resp_power + EPS) / (motion_power + EPS))
    quality = float(1.0 / (1.0 + math.exp(-snr_db / 6.0)))
    return quality, snr_db, motion_power


def estimate_surrogate(
    surrogate: np.ndarray,
    eigvals: np.ndarray,
    pca_ratio: float,
    fs_hz: float,
    cfg: Config,
    timestamp_quality: float = 1.0,
    movement_quality: float = 1.0,
) -> tuple[Estimate, dict[str, Any], dict[str, Any]]:
    psd = psd_estimate(surrogate, fs_hz, cfg)
    acf = autocorrelation_estimate(surrogate, fs_hz, cfg)
    rr_final, consensus_score, agreement = consensus_estimate(psd, acf, cfg)

    # Confidence is a transparent heuristic; all ingredients are exported so it
    # can later be recalibrated against a true respiratory reference.
    confidence = (
        0.25 * psd["quality"]
        + 0.20 * acf["quality"]
        + 0.15 * pca_ratio
        + 0.20 * agreement
        + 0.10 * timestamp_quality
        + 0.10 * movement_quality
    )
    confidence *= 0.75 + 0.25 * consensus_score
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # Mark harmonic resolution even when the purely spectral pre-check did not
    # fire but the PSD-vs-ACF consensus safely mapped an integer harmonic back
    # to its fundamental.
    consensus_harmonic_resolution = False
    rr_psd = float(psd.get("rr_bpm", float("nan")))
    rr_acf = float(acf.get("rr_bpm", float("nan")))
    if np.isfinite(rr_psd) and np.isfinite(rr_acf) and np.isfinite(rr_final):
        for multiple in (2.0, 3.0):
            if abs(rr_psd / multiple - rr_final) <= cfg.consensus_agreement_bpm and abs(
                rr_acf - rr_final
            ) <= cfg.consensus_agreement_bpm:
                consensus_harmonic_resolution = True
                break

    est = Estimate(
        rr_psd_raw_bpm=float(psd["rr_raw_bpm"]),
        rr_psd_bpm=float(psd["rr_bpm"]),
        rr_acf_bpm=float(acf["rr_bpm"]),
        rr_final_bpm=float(rr_final),
        spectral_quality=float(psd["quality"]),
        acf_quality=float(acf["quality"]),
        agreement_quality=float(agreement),
        consensus_score=float(consensus_score),
        harmonic_correction_applied=bool(
            psd["harmonic_corrected"] or consensus_harmonic_resolution
        ),
        pca_pc1_variance_ratio=float(pca_ratio),
        pca_eigenvalues=[float(v) for v in eigvals],
        confidence=confidence,
    )
    return est, psd, acf


def kalman_track_rr(
    measurements: np.ndarray,
    confidence: np.ndarray,
    hop_s: float,
    cfg: Config,
) -> np.ndarray:
    out = np.full(len(measurements), np.nan, dtype=float)
    x = float("nan")
    p = 25.0
    q = cfg.kalman_process_var_bpm2_per_s * hop_s

    for i, (z, c) in enumerate(zip(measurements, confidence)):
        if np.isfinite(x):
            p += q
        valid = np.isfinite(z) and c >= cfg.min_confidence
        if valid:
            if not np.isfinite(x):
                x = float(z)
                p = cfg.kalman_base_measurement_var_bpm2 / max(c * c, 0.05)
            else:
                r = cfg.kalman_base_measurement_var_bpm2 / max(c * c, 0.05)
                k = p / (p + r)
                x = x + k * (float(z) - x)
                p = (1.0 - k) * p
        if np.isfinite(x):
            out[i] = x
    return out
