"""
Array signal model for Uniform Linear Array (ULA).

X = A*S + N
  A : M x D steering matrix
  S : D x N snapshot matrix
  N : M x N noise matrix
"""

import numpy as np


def steering_vector(theta_deg: float, M: int, d: float = 0.5) -> np.ndarray:
    """
    Steering vector for a single angle.

    Parameters
    ----------
    theta_deg : angle in degrees
    M         : number of array elements
    d         : inter-element spacing in wavelengths (default 0.5)

    Returns
    -------
    a : (M,) complex array
    """
    theta = np.deg2rad(theta_deg)
    m = np.arange(M)
    return np.exp(1j * 2 * np.pi * d * m * np.sin(theta))


def steering_matrix(thetas_deg: list, M: int, d: float = 0.5) -> np.ndarray:
    """
    Steering matrix A for multiple angles.

    Returns
    -------
    A : (M, D) complex array
    """
    return np.column_stack([steering_vector(t, M, d) for t in thetas_deg])


def generate_received_signal(
    thetas_deg: list,
    snrs_db: list,
    M: int,
    N_snapshots: int,
    coherent: bool = False,
    noise_power: float = 1.0,
    d: float = 0.5,
    seed: int = None,
) -> np.ndarray:
    """
    Generate received array signal X = A*S + N.

    Parameters
    ----------
    thetas_deg   : list of true DOAs in degrees
    snrs_db      : list of per-source SNR in dB (same length as thetas_deg)
    M            : number of array elements
    N_snapshots  : number of time snapshots
    coherent     : if True, all sources are fully coherent (same waveform)
    noise_power  : variance of additive white Gaussian noise
    d            : inter-element spacing in wavelengths
    seed         : random seed for reproducibility

    Returns
    -------
    X : (M, N_snapshots) complex received signal matrix
    """
    rng = np.random.default_rng(seed)
    D = len(thetas_deg)
    A = steering_matrix(thetas_deg, M, d)

    signal_powers = [noise_power * 10 ** (snr / 10) for snr in snrs_db]

    if coherent:
        base_signal = (
            np.sqrt(signal_powers[0]) * (rng.standard_normal((1, N_snapshots)) +
            1j * rng.standard_normal((1, N_snapshots))) / np.sqrt(2)
        )
        S = np.vstack([base_signal for _ in range(D)])
    else:
        S = np.vstack([
            np.sqrt(p) * (rng.standard_normal((1, N_snapshots)) +
                          1j * rng.standard_normal((1, N_snapshots))) / np.sqrt(2)
            for p in signal_powers
        ])

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal((M, N_snapshots)) +
        1j * rng.standard_normal((M, N_snapshots))
    )

    return A @ S + noise


def sample_covariance(X: np.ndarray) -> np.ndarray:
    """
    Sample covariance matrix R = (1/N) * X @ X^H.

    Returns
    -------
    R : (M, M) Hermitian matrix
    """
    N = X.shape[1]
    return (X @ X.conj().T) / N


def spatial_smoothing_covariance(X: np.ndarray, L: int) -> np.ndarray:
    """
    Forward spatial smoothing covariance matrix.

    Divides the M-element array into L overlapping subarrays of size (M-L+1),
    then averages their covariance matrices.

    Parameters
    ----------
    X : (M, N) received signal
    L : number of subarrays

    Returns
    -------
    R_smooth : (M-L+1, M-L+1) smoothed covariance matrix
    """
    M = X.shape[0]
    N_sub = M - L + 1  # subarray size
    R_sum = np.zeros((N_sub, N_sub), dtype=complex)
    for l in range(L):
        X_sub = X[l:l + N_sub, :]
        R_sum += sample_covariance(X_sub)
    return R_sum / L


def estimate_snr(X: np.ndarray, D: int) -> float:
    """
    Rough SNR estimate from eigenvalue spread of sample covariance matrix.

    Uses the ratio of the D-th largest eigenvalue to the noise floor
    (mean of the M-D smallest eigenvalues).

    Parameters
    ----------
    X : (M, N) received signal
    D : assumed number of sources

    Returns
    -------
    snr_db : estimated SNR in dB
    """
    R = sample_covariance(X)
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(R)))[::-1]
    M = R.shape[0]
    noise_floor = np.mean(eigenvalues[D:])
    signal_ev = eigenvalues[D - 1]
    if noise_floor < 1e-12:
        return 40.0
    return 10 * np.log10(signal_ev / noise_floor)
