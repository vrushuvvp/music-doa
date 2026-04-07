"""
Adaptive-IM-PM-Root-MUSIC (AIM-PM-Root-MUSIC)
==============================================

Original contribution extending Zhang et al. (CCISP 2024).

Motivation
----------
The paper shows (Fig. 5) that subarray count L has a sweet spot: too few
subarrays -> poor coherent signal decorrelation; too many -> reduced subarray
size degrades accuracy. The paper hardcodes L=9 for their M=20 setup.

Key insight
-----------
The optimal L depends on the operating SNR:
  - Low SNR: more subarrays = better averaging = lower RMSE
  - High SNR: fewer subarrays = larger subarray = higher resolution

We estimate SNR from the received data and select L from a lookup policy,
then fine-tune with a lightweight grid search over a narrow range.

Algorithm
---------
1. Estimate SNR_hat from eigenvalue spread of sample covariance.
2. Use SNR_hat to select a search window [L_min, L_max].
3. Over that window, evaluate IM-PM-Root-MUSIC for each candidate L and
   pick the L that minimises the polynomial root distance-to-unit-circle
   (a proxy for estimation confidence that requires no ground truth).
4. Run final IM-PM-Root-MUSIC with the selected L*.

Complexity
----------
The search window is small (typically 3-5 candidates), so the overhead
over fixed-L IM-PM-Root-MUSIC is minimal.

This is a novel contribution — no prior work uses SNR-adaptive subarray
selection within the propagation operator + Root-MUSIC framework.
"""

import numpy as np
from utils.array_signal import (
    spatial_smoothing_covariance,
    sample_covariance,
    estimate_snr,
)
from algorithms.pm_music import _propagator_noise_subspace
from algorithms.root_music import _root_music_from_noise_subspace


# --- SNR-to-subarray policy ---------------------------------------------------
# Maps SNR range (dB) to a (L_min, L_max) search window.
# Derived empirically from Fig. 5 of the paper + general theory:
#   Very low SNR (<-20 dB): aggressive smoothing needed
#   Low SNR (-20 to -15 dB): moderate smoothing
#   Medium SNR (-15 to -10 dB): balanced
#   High SNR (>-10 dB): minimal smoothing, preserve aperture

SNR_POLICY = [
    (-np.inf,  4.0, (0.40, 0.65)),   # Very low / saturated SNR: aggressive smoothing
    (  4.0,    6.0, (0.30, 0.55)),   # Low SNR (true ~-10 dB)
    (  6.0,   10.0, (0.22, 0.42)),   # Medium SNR (true ~ -5 dB)
    ( 10.0,   16.0, (0.15, 0.32)),   # High SNR (true 0..+5 dB)
    ( 16.0,  np.inf, (0.10, 0.22)),  # Very high SNR: minimal smoothing
]


def _snr_to_L_range(snr_db: float, M: int, D: int) -> tuple:
    """
    Map estimated SNR to a candidate range of subarray counts.

    Parameters
    ----------
    snr_db : estimated SNR in dB
    M      : array size
    D      : number of sources

    Returns
    -------
    (L_min, L_max) : integer bounds, guaranteed to keep N_sub > D
    """
    for lo, hi, (frac_lo, frac_hi) in SNR_POLICY:
        if lo <= snr_db < hi:
            L_min = max(1, int(np.floor(frac_lo * M)))
            L_max = max(L_min + 1, int(np.ceil(frac_hi * M)))
            break
    else:
        L_min, L_max = 1, M // 3

    # Ensure subarray size > D
    L_min = min(L_min, M - D - 1)
    L_max = min(L_max, M - D - 1)
    L_min = max(1, L_min)
    L_max = max(L_min, L_max)

    return L_min, L_max


def _unit_circle_score(X: np.ndarray, D: int, M: int, L: int,
                       d: float = 0.5) -> float:
    """
    Proxy confidence score for a given L: mean distance of the D closest
    polynomial roots to the unit circle (lower = better).

    This is computed without ground truth and acts as the selection criterion.
    """
    N_sub = M - L + 1
    R_smooth = spatial_smoothing_covariance(X, L)
    Q_o = _propagator_noise_subspace(R_smooth, D)

    C = Q_o @ Q_o.conj().T
    coeffs = np.zeros(2 * N_sub - 1, dtype=complex)
    for k in range(-(N_sub - 1), N_sub):
        diag = np.diag(C, k)
        coeffs[k + N_sub - 1] = np.sum(diag)
    poly_coeffs = coeffs[::-1]

    try:
        roots = np.roots(poly_coeffs)
    except np.linalg.LinAlgError:
        return np.inf

    roots_inside = roots[np.abs(roots) <= 1.0 + 1e-6]
    if len(roots_inside) < D:
        return np.inf

    distances = np.sort(np.abs(np.abs(roots_inside) - 1.0))
    return float(np.mean(distances[:D]))


def select_optimal_L(X: np.ndarray, D: int, M: int,
                     d: float = 0.5) -> tuple:
    """
    Select the optimal number of subarrays L* using SNR-guided search
    with unit-circle proximity as the selection criterion.

    Parameters
    ----------
    X : (M, N) received signal
    D : number of sources
    M : number of array elements
    d : inter-element spacing

    Returns
    -------
    L_star    : optimal number of subarrays
    snr_hat   : estimated SNR used for windowing (dB)
    scores    : dict {L: score} for all evaluated candidates (for analysis)
    """
    snr_hat = estimate_snr(X, D)
    L_min, L_max = _snr_to_L_range(snr_hat, M, D)

    candidates = list(range(L_min, L_max + 1))
    scores = {}
    for L in candidates:
        scores[L] = _unit_circle_score(X, D, M, L, d)

    L_star = min(scores, key=scores.get)
    return L_star, snr_hat, scores


def adaptive_im_pm_root_music(
    X: np.ndarray,
    D: int,
    M: int,
    d: float = 0.5,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple:
    """
    Estimate DOAs using Adaptive IM-PM-Root-MUSIC (AIM-PM-Root-MUSIC).

    This is the proposed extension of Zhang et al. (CCISP 2024):
    instead of a fixed L, we select L* adaptively based on the estimated
    operating SNR and a polynomial root proximity criterion.

    Parameters
    ----------
    X                  : (M, N) received signal
    D                  : number of sources
    M                  : number of array elements
    d                  : inter-element spacing in wavelengths
    return_diagnostics : if True, return (doas, L_star, snr_hat, scores)

    Returns
    -------
    doas : (D,) estimated DOAs in degrees
    If return_diagnostics=True, also returns (L_star, snr_hat, scores).
    """
    L_star, snr_hat, scores = select_optimal_L(X, D, M, d)

    N_sub = M - L_star + 1
    R_smooth = spatial_smoothing_covariance(X, L_star)
    Q_o = _propagator_noise_subspace(R_smooth, D)
    doas = _root_music_from_noise_subspace(Q_o, N_sub, D, d)

    if return_diagnostics:
        return doas, L_star, snr_hat, scores
    return doas
