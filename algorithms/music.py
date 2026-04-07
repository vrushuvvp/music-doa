"""
Classical MUSIC algorithm for DOA estimation.

Reference: Schmidt, R. O. (1986). Multiple emitter location and signal
parameter estimation. IEEE Transactions on Antennas and Propagation, 34(3).
"""

import numpy as np
from utils.array_signal import sample_covariance, steering_vector


def music(X: np.ndarray, D: int, M: int, d: float = 0.5,
          angle_range: tuple = (-90, 90), angle_step: float = 0.1) -> np.ndarray:
    """
    Estimate DOAs using the classical MUSIC algorithm (spectral peak search).

    Parameters
    ----------
    X           : (M, N) received signal matrix
    D           : number of sources
    M           : number of array elements
    d           : inter-element spacing in wavelengths
    angle_range : (min, max) search range in degrees
    angle_step  : angular resolution in degrees

    Returns
    -------
    doas : (D,) estimated DOAs in degrees
    """
    R = sample_covariance(X)
    eigenvalues, eigenvectors = np.linalg.eigh(R)

    # Noise subspace: eigenvectors corresponding to M-D smallest eigenvalues
    # eigh returns in ascending order
    G_n = eigenvectors[:, :M - D]  # (M, M-D)

    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    spectrum = np.zeros(len(angles))

    for i, theta in enumerate(angles):
        a = steering_vector(theta, M, d)
        proj = a.conj() @ G_n @ G_n.conj().T @ a
        spectrum[i] = 1.0 / np.abs(proj)

    # Find D largest peaks
    doas = _find_peaks(angles, spectrum, D)
    return np.array(doas)


def _find_peaks(angles: np.ndarray, spectrum: np.ndarray, D: int) -> list:
    """Find D largest local maxima in the spectrum."""
    peaks = []
    for i in range(1, len(spectrum) - 1):
        if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
            peaks.append((spectrum[i], angles[i]))
    peaks.sort(reverse=True)
    return [p[1] for p in peaks[:D]]
