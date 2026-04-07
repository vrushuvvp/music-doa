"""
Simulation 1: Estimation of multiple non-coherent signals.
Replicates Fig. 2 from Zhang et al. (CCISP 2024).

Setup:
  - 3 sources at -30°, -50°, -10°
  - SNRs: 0 dB, -5 dB, -10 dB
  - M=20 elements, N=100 snapshots, d=0.5λ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.array_signal import generate_received_signal, steering_vector
from utils.array_signal import sample_covariance, spatial_smoothing_covariance
from algorithms.pm_music import _propagator_noise_subspace

# ── Simulation parameters ────────────────────────────────────────────────────
THETAS = [-30.0, -50.0, -10.0]
SNRS_DB = [0.0, -5.0, -10.0]
M = 20
N_SNAPSHOTS = 100
D = len(THETAS)
d = 0.5
ANGLE_RANGE = (-90, 90)
ANGLE_STEP = 0.5

SEED = 42

# ── Generate signal ───────────────────────────────────────────────────────────
X = generate_received_signal(THETAS, SNRS_DB, M, N_SNAPSHOTS,
                              coherent=False, seed=SEED)

angles = np.arange(ANGLE_RANGE[0], ANGLE_RANGE[1] + ANGLE_STEP, ANGLE_STEP)

# ── Helper: compute MUSIC spectrum from noise subspace ────────────────────────
def music_spectrum(G_n, M_local, angles, d=0.5):
    spec = []
    for theta in angles:
        a = steering_vector(theta, M_local, d)
        proj = a.conj() @ G_n @ G_n.conj().T @ a
        spec.append(1.0 / (np.abs(proj) + 1e-12))
    spec = np.array(spec)
    return spec / spec.max()


# ── Classical MUSIC ───────────────────────────────────────────────────────────
R = sample_covariance(X)
_, evec = np.linalg.eigh(R)
G_n_music = evec[:, :M - D]
spec_music = music_spectrum(G_n_music, M, angles)

# ── PM-MUSIC ──────────────────────────────────────────────────────────────────
Q_pm = _propagator_noise_subspace(R, D)
spec_pm = music_spectrum(Q_pm, M, angles)

# ── Root-MUSIC (convert roots to pseudo-spectrum for visualisation) ───────────
from algorithms.root_music import root_music
doas_root = root_music(X, D, M, d)
# Represent as impulses for comparison
spec_root = np.zeros(len(angles))
for doa in doas_root:
    idx = np.argmin(np.abs(angles - doa))
    spec_root[idx] = 1.0

# ── PM-Root-MUSIC ─────────────────────────────────────────────────────────────
from algorithms.pm_root_music import pm_root_music
doas_pm_root = pm_root_music(X, D, M, d)
spec_pm_root = np.zeros(len(angles))
for doa in doas_pm_root:
    idx = np.argmin(np.abs(angles - doa))
    spec_pm_root[idx] = 1.0

# ── IM-PM-Root-MUSIC ──────────────────────────────────────────────────────────
from algorithms.im_pm_root_music import im_pm_root_music
doas_im = im_pm_root_music(X, D, M, L=9, d=d)
spec_im = np.zeros(len(angles))
for doa in doas_im:
    idx = np.argmin(np.abs(angles - doa))
    spec_im[idx] = 1.0

# ── AIM-PM-Root-MUSIC (our contribution) ─────────────────────────────────────
from algorithms.adaptive_im_pm_root_music import adaptive_im_pm_root_music
doas_aim, L_star, snr_hat, _ = adaptive_im_pm_root_music(X, D, M, d,
                                                          return_diagnostics=True)
spec_aim = np.zeros(len(angles))
for doa in doas_aim:
    idx = np.argmin(np.abs(angles - doa))
    spec_aim[idx] = 1.0

print(f"Estimated SNR: {snr_hat:.1f} dB, Selected L*: {L_star}")
print(f"True DOAs: {sorted(THETAS)}")
print(f"AIM-PM-Root-MUSIC DOAs: {sorted(doas_aim.tolist())}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(angles, spec_music,  label='MUSIC',                linewidth=1.5)
ax.plot(angles, spec_pm,     label='PM-MUSIC',             linewidth=1.5, linestyle='--')
ax.stem(angles, spec_root,   label='Root-MUSIC',           linefmt='g-',
        markerfmt='go', basefmt=' ')
ax.stem(angles, spec_pm_root, label='PM-Root-MUSIC',       linefmt='r-',
        markerfmt='rs', basefmt=' ')
ax.stem(angles, spec_im,      label='IM-PM-Root-MUSIC',    linefmt='m-',
        markerfmt='m^', basefmt=' ')
ax.stem(angles, spec_aim,     label='AIM-PM-Root-MUSIC',   linefmt='k-',
        markerfmt='kD', basefmt=' ')

for t in THETAS:
    ax.axvline(t, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

ax.set_xlabel('Incident angle (°)', fontsize=12)
ax.set_ylabel('Normalised amplitude', fontsize=12)
ax.set_title('Non-coherent signal estimation (replicates Fig. 2)', fontsize=13)
ax.legend(fontsize=9)
ax.set_xlim(-90, 90)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/sim1_noncoherent.png', dpi=150, bbox_inches='tight')
print("Saved: results/sim1_noncoherent.png")
plt.show()
