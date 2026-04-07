"""
PM-MUSIC: MUSIC with noise subspace estimated via the Propagation operator.

Avoids eigenvalue decomposition — replaces it with a pseudo-inverse solve.
Complexity: O(NM^2) for covariance + O(M*D^2) for orthogonalization,
            + O(n*M*D) for spectral search.

Reference: Marcos, S., Marsal, A., & Benidir, M. (1995).
The propagator method for source bearing estimation.
Signal Processing, 42(2), 121-138.
"""

import numpy as np
from utils.array_signal import sample_covariance, steering_vector


def _propagator_noise_subspace(R: np.ndarray, D: int) -> np.ndarray:
    """
    Estimate the noise subspace matrix Q using the propagation operator.

    The covariance matrix R is partitioned as R = [G | H]
    where G is the first D columns and H is the remaining M-D columns.

    P = (G^H G)^{-1} G^H H   (propagation operator)
    Q = [-P^H | I_{M-D}]^H   (noise subspace basis, unnormalized)
    Q_o = Q (Q^H Q)^{-1/2}   (orthonormalized)

    Returns
    -------
    Q_o : (M, M-D) orthonormal noise subspace matrix
    """
    M = R.shape[0]
    G = R[:, :D]        # (M, D)
    H = R[:, D:]        # (M, M-D)

    # Propagation operator: P = pinv(G) @ H  ->  (D, M-D)
    P = np.linalg.pinv(G) @ H  # (D, M-D)

    # Build Q: [−P^H ; I_{M-D}]  -> shape (M, M-D)
    # -P.conj().T has shape (M-D, D) — WRONG direction
    # We need Q such that A^H Q = 0, where A is (M, D)
    # Standard: Q = [-P^H ; I_{M-D}]  means top block is (D, M-D) = -P
    # and bottom block is (M-D, M-D) = I
    # So Q shape is (M, M-D): top D rows = -P, bottom M-D rows = I
    Q = np.vstack([-P, np.eye(M - D, dtype=complex)])  # (M, M-D)

    # Orthonormalize Q
    QhQ = Q.conj().T @ Q
    eigvals, eigvecs = np.linalg.eigh(QhQ)
    # QhQ^{-1/2} = V * diag(1/sqrt(lambda)) * V^H
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.abs(eigvals) + 1e-12)) @ eigvecs.conj().T
    Q_o = Q @ inv_sqrt
    return Q_o


def pm_music(X: np.ndarray, D: int, M: int, d: float = 0.5,
             angle_range: tuple = (-90, 90), angle_step: float = 0.1) -> np.ndarray:
    """
    Estimate DOAs using PM-MUSIC (propagation operator + spectral search).

    Parameters
    ----------
    X           : (M, N) received signal matrix
    D           : number of sources
    M           : number of array elements
    d           : inter-element spacing in wavelengths
    angle_range : (min_deg, max_deg) search range
    angle_step  : angular resolution in degrees

    Returns
    -------
    doas : (D,) estimated DOAs in degrees
    """
    R = sample_covariance(X)
    Q_o = _propagator_noise_subspace(R, D)

    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    spectrum = np.zeros(len(angles))

    for i, theta in enumerate(angles):
        a = steering_vector(theta, M, d)
        proj = a.conj() @ Q_o @ Q_o.conj().T @ a
        spectrum[i] = 1.0 / (np.abs(proj) + 1e-12)

    from algorithms.music import _find_peaks
    return np.array(_find_peaks(angles, spectrum, D))
