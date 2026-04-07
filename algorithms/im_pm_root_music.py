"""
IM-PM-Root-MUSIC: the algorithm proposed in the IEEE paper.

Enhancement over PM-Root-MUSIC:
  Replaces the standard sample covariance matrix with a spatially smoothed
  covariance matrix, enabling estimation of coherent signals.

Pipeline:
  1. Forward spatial smoothing -> R_smooth  (decorrelates coherent sources)
  2. Propagation operator -> noise subspace Q_o  (no eigendecomposition)
  3. Root-MUSIC polynomial rooting -> DOAs  (no spectral search)

Reference: Zhang et al., "A Fast and Robust MUSIC Algorithm for Estimating
Multiple Coherent Signals", CCISP 2024, DOI: 10.1109/CCISP63826.2024.10765572
"""

import numpy as np
from utils.array_signal import spatial_smoothing_covariance
from algorithms.pm_music import _propagator_noise_subspace
from algorithms.root_music import _root_music_from_noise_subspace


def im_pm_root_music(X: np.ndarray, D: int, M: int,
                     L: int = 9, d: float = 0.5) -> np.ndarray:
    """
    Estimate DOAs using IM-PM-Root-MUSIC (fixed subarray count).

    Parameters
    ----------
    X : (M, N) received signal
    D : number of sources
    M : number of array elements
    L : number of subarrays for spatial smoothing (paper recommends 9 for M=20)
    d : inter-element spacing in wavelengths

    Returns
    -------
    doas : (D,) estimated DOAs in degrees

    Notes
    -----
    Subarray size: N_sub = M - L + 1
    Constraint: N_sub > D (subarray must have more elements than sources)
    """
    N_sub = M - L + 1
    assert N_sub > D, (
        f"Subarray size ({N_sub}) must exceed number of sources ({D}). "
        f"Reduce L or increase M."
    )

    R_smooth = spatial_smoothing_covariance(X, L)
    Q_o = _propagator_noise_subspace(R_smooth, D)

    return _root_music_from_noise_subspace(Q_o, N_sub, D, d)
