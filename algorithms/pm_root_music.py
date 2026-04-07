"""
PM-Root-MUSIC: combines propagation operator noise subspace estimation
with Root-MUSIC polynomial rooting.

Complexity vs classical MUSIC:
  - Removes O(M^3) eigendecomposition -> O(M*D^2) orthogonalization
  - Removes O(n*M*D) spectral search -> O(M^3) polynomial rooting (but M is now
    smaller after spatial smoothing, so total cost drops significantly)
"""

import numpy as np
from utils.array_signal import sample_covariance
from algorithms.pm_music import _propagator_noise_subspace
from algorithms.root_music import _root_music_from_noise_subspace


def pm_root_music(X: np.ndarray, D: int, M: int, d: float = 0.5) -> np.ndarray:
    """
    Estimate DOAs using PM-Root-MUSIC.

    Parameters
    ----------
    X : (M, N) received signal
    D : number of sources
    M : number of array elements
    d : inter-element spacing in wavelengths

    Returns
    -------
    doas : (D,) estimated DOAs in degrees
    """
    R = sample_covariance(X)
    Q_o = _propagator_noise_subspace(R, D)

    return _root_music_from_noise_subspace(Q_o, M, D, d)
