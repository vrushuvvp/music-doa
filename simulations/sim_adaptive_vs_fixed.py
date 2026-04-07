"""
Simulation: Adaptive L* selection vs fixed-L IM-PM-Root-MUSIC.
This is the key experiment for the contribution.

Shows that AIM-PM-Root-MUSIC with adaptive L*:
  - Matches or beats L=9 at low SNR (better smoothing)
  - Beats L=9 at high SNR (preserved aperture)
  - Tracks the oracle-best L curve closely

Also plots: distribution of L* choices across SNR levels.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.array_signal import generate_received_signal
from utils.metrics import rmse

from algorithms.im_pm_root_music import im_pm_root_music
from algorithms.adaptive_im_pm_root_music import adaptive_im_pm_root_music

# ── Parameters ────────────────────────────────────────────────────────────────
TRUE_DOA = 20.0
M = 20
D = 1
N_SNAPSHOTS = 100
d = 0.5
N_TRIALS = 100          # increase to 500+ for publication
SNR_RANGE = np.arange(-25, -9, 2)

FIXED_L_VALUES = [3, 6, 9, 13]  # L values to compare (matches Fig. 5 in paper)

print(f"Running {N_TRIALS} trials × {len(SNR_RANGE)} SNR points...")

rmse_fixed = {L: [] for L in FIXED_L_VALUES}
rmse_adaptive = []
oracle_rmse = []
mean_L_star = []

for snr in SNR_RANGE:
    print(f"  SNR = {snr:+.0f} dB", flush=True)

    # Fixed-L results
    fixed_trials = {L: [] for L in FIXED_L_VALUES}
    adaptive_trials = []
    L_star_trials = []

    for trial in range(N_TRIALS):
        X = generate_received_signal([TRUE_DOA], [snr], M, N_SNAPSHOTS,
                                     coherent=False, seed=trial + snr * 1000)
        for L in FIXED_L_VALUES:
            try:
                est = im_pm_root_music(X, D, M, L=L, d=d)[0]
                fixed_trials[L].append(float(est))
            except Exception:
                fixed_trials[L].append(TRUE_DOA + 90.0)

        try:
            est_a, L_star, _, _ = adaptive_im_pm_root_music(
                X, D, M, d, return_diagnostics=True)
            adaptive_trials.append(float(est_a[0]))
            L_star_trials.append(L_star)
        except Exception:
            adaptive_trials.append(TRUE_DOA + 90.0)
            L_star_trials.append(9)

    for L in FIXED_L_VALUES:
        rmse_fixed[L].append(rmse(np.array(fixed_trials[L]), TRUE_DOA))

    rmse_adaptive.append(rmse(np.array(adaptive_trials), TRUE_DOA))
    mean_L_star.append(np.mean(L_star_trials))

    # Oracle: which fixed-L performed best at this SNR?
    oracle_rmse.append(min(rmse_fixed[L][-1] for L in FIXED_L_VALUES))

# ── Plot 1: RMSE comparison ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
line_styles = {3: ('o--', '#4c7bf3'), 6: ('s--', '#f3924c'),
               9: ('^--', '#4caf50'), 13: ('D--', '#f34c7b')}

for L in FIXED_L_VALUES:
    mk, col = line_styles[L]
    ax.plot(SNR_RANGE, rmse_fixed[L], mk, label=f'Fixed L={L}',
            color=col, linewidth=1.5, markersize=6)

ax.plot(SNR_RANGE, oracle_rmse, 'k:', linewidth=1.5, label='Oracle best-L', alpha=0.6)
ax.plot(SNR_RANGE, rmse_adaptive, 'k*-', linewidth=2.5, markersize=10,
        label='AIM-PM-Root-MUSIC (adaptive L*)', zorder=5)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('RMSE (degrees)', fontsize=12)
ax.set_title('Fixed-L vs Adaptive-L (AIM-PM-Root-MUSIC)', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 2: Adaptive L* distribution ─────────────────────────────────────────
ax2 = axes[1]
ax2.plot(SNR_RANGE, mean_L_star, 'k*-', linewidth=2, markersize=10)
ax2.fill_between(SNR_RANGE,
                 [l - 1 for l in mean_L_star],
                 [l + 1 for l in mean_L_star],
                 alpha=0.2, color='gray', label='±1 subarray')
ax2.axhline(9, color='green', linestyle='--', alpha=0.6, label='Paper\'s fixed L=9')
ax2.set_xlabel('SNR (dB)', fontsize=12)
ax2.set_ylabel('Mean selected L*', fontsize=12)
ax2.set_title('SNR-adaptive subarray selection', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, M)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/sim_adaptive_vs_fixed.png', dpi=150, bbox_inches='tight')
print("Saved: results/sim_adaptive_vs_fixed.png")
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── AIM-PM-Root-MUSIC vs IM-PM-Root-MUSIC (L=9) ──")
print(f"{'SNR':>8} {'Fixed L=9 RMSE':>18} {'Adaptive RMSE':>18} {'Mean L*':>10} {'Gain (dB)':>12}")
for i, snr in enumerate(SNR_RANGE):
    fixed9 = rmse_fixed[9][i]
    adap = rmse_adaptive[i]
    gain = 20 * np.log10(fixed9 / adap) if adap > 1e-6 else 0.0
    print(f"{snr:>8.0f} {fixed9:>18.4f} {adap:>18.4f} "
          f"{mean_L_star[i]:>10.1f} {gain:>12.2f}")
