"""
Root-MUSIC algorithm.

Converts spectral peak search into a polynomial root-finding problem,
reducing complexity from O(n*M*D) search to O(M^3) polynomial rooting.

Reference: Barabell, A. (1983). Improving the resolution performance of
eigenstructure-based direction-finding algorithms. ICASSP.
"""

import numpy as np
from utils.array_signal import sample_covariance


def _root_music_from_noise_subspace(G_n: np.ndarray, M: int,
                                     D: int, d: float = 0.5) -> np.ndarray:
    """
    Core Root-MUSIC: find roots of the polynomial C(z) = p(z)^H * U_n * U_n^H * p(z).

    The polynomial has degree 2*(M-1). We look for D roots inside/near the unit circle.

    Parameters
    ----------
    G_n : (M, M-D) noise subspace matrix
    M   : number of array elements
    D   : number of sources
    d   : inter-element spacing in wavelengths

    Returns
    -------
    doas : (D,) estimated DOAs in degrees
    """
    # C = G_n @ G_n^H, polynomial coefficient matrix
    C = G_n @ G_n.conj().T  # (M, M)

    # Build polynomial coefficients by summing diagonals of C
    # f(z) = sum_{k=-(M-1)}^{M-1} c_k * z^k where c_k = sum of k-th diagonal
    coeffs = np.zeros(2 * M - 1, dtype=complex)
    for k in range(-(M - 1), M):
        diag = np.diag(C, k)
        coeffs[k + M - 1] = np.sum(diag)

    # coeffs[0] is z^{-(M-1)}, multiply through by z^{M-1} to get standard poly
    # Polynomial order: 2*(M-1), highest power first
    poly_coeffs = coeffs[::-1]  # numpy poly convention: highest power first

    roots = np.roots(poly_coeffs)

    # Select D roots closest to the unit circle (but inside it)
    # Exclude roots exactly on or outside the unit circle
    roots_inside = roots[np.abs(roots) <= 1.0 + 1e-6]
    distances = np.abs(np.abs(roots_inside) - 1.0)
    idx = np.argsort(distances)[:D]
    selected_roots = roots_inside[idx]

    # Convert roots to angles: z = exp(j*2*pi*d*sin(theta))
    doas = []
    for z in selected_roots:
        angle_arg = np.angle(z) / (2 * np.pi * d)
        angle_arg = np.clip(angle_arg, -1.0, 1.0)
        doas.append(np.rad2deg(np.arcsin(angle_arg)))

    return np.array(sorted(doas))


def root_music(X: np.ndarray, D: int, M: int, d: float = 0.5) -> np.ndarray:
    """
    Estimate DOAs using classical Root-MUSIC (eigendecomposition + polynomial rooting).

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
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    G_n = eigenvectors[:, :M - D]  # noise subspace

    return _root_music_from_noise_subspace(G_n, M, D, d)
