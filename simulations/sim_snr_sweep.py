"""
Simulation: RMSE vs SNR sweep with 1000 Monte Carlo trials.
Replicates Fig. 4 from the paper + adds AIM-PM-Root-MUSIC comparison.

Setup: single source at 20°, M=20 elements, N=100 snapshots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from utils.array_signal import generate_received_signal
from utils.metrics import rmse

from algorithms.music import music
from algorithms.pm_music import pm_music
from algorithms.root_music import root_music
from algorithms.pm_root_music import pm_root_music
from algorithms.im_pm_root_music import im_pm_root_music
from algorithms.adaptive_im_pm_root_music import adaptive_im_pm_root_music

# ── Parameters ────────────────────────────────────────────────────────────────
TRUE_DOA = 20.0
M = 20
D = 1
N_SNAPSHOTS = 100
d = 0.5
N_TRIALS = 200          # use 1000 for publication; 200 for quick testing
SNR_RANGE = np.arange(-25, -14, 1)

algorithms_cfg = {
    'MUSIC':              lambda X: music(X, D, M, d)[0],
    'PM-MUSIC':           lambda X: pm_music(X, D, M, d)[0],
    'Root-MUSIC':         lambda X: root_music(X, D, M, d)[0],
    'PM-Root-MUSIC':      lambda X: pm_root_music(X, D, M, d)[0],
    'IM-PM-Root-MUSIC':   lambda X: im_pm_root_music(X, D, M, L=9, d=d)[0],
    'AIM-PM-Root-MUSIC':  lambda X: adaptive_im_pm_root_music(X, D, M, d)[0],
}

rmse_results = {name: [] for name in algorithms_cfg}

print(f"Running {N_TRIALS} Monte Carlo trials per SNR point ({len(SNR_RANGE)} points)...")
for snr in SNR_RANGE:
    print(f"  SNR = {snr:+.0f} dB", flush=True)
    trial_results = {name: [] for name in algorithms_cfg}
    for trial in range(N_TRIALS):
        X = generate_received_signal([TRUE_DOA], [snr], M, N_SNAPSHOTS,
                                     coherent=False, seed=trial)
        for name, fn in algorithms_cfg.items():
            try:
                est = fn(X)
                trial_results[name].append(float(est))
            except Exception:
                trial_results[name].append(TRUE_DOA + 90.0)  # penalty

    for name in algorithms_cfg:
        rmse_results[name].append(rmse(np.array(trial_results[name]), TRUE_DOA))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

styles = {
    'MUSIC':             ('o-',  'tab:blue'),
    'PM-MUSIC':          ('s--', 'tab:orange'),
    'Root-MUSIC':        ('^-',  'tab:green'),
    'PM-Root-MUSIC':     ('D--', 'tab:red'),
    'IM-PM-Root-MUSIC':  ('v-',  'tab:purple'),
    'AIM-PM-Root-MUSIC': ('*-',  'black'),
}

for name, vals in rmse_results.items():
    marker, color = styles[name]
    lw = 2.5 if 'AIM' in name else 1.5
    ax.plot(SNR_RANGE, vals, marker, label=name, color=color,
            linewidth=lw, markersize=6 if 'AIM' not in name else 9)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('RMSE (degrees)', fontsize=12)
ax.set_title(f'RMSE vs SNR ({N_TRIALS} Monte Carlo trials, source at {TRUE_DOA}°)',
             fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/sim_snr_sweep.png', dpi=150, bbox_inches='tight')
print("Saved: results/sim_snr_sweep.png")
plt.show()

# ── Print summary table ───────────────────────────────────────────────────────
print("\n── RMSE at selected SNR points ──────────────────────────────────")
header = f"{'SNR (dB)':>10}" + "".join(f"{n:>22}" for n in algorithms_cfg)
print(header)
for i, snr in enumerate(SNR_RANGE):
    row = f"{snr:>10.0f}" + "".join(
        f"{rmse_results[n][i]:>22.4f}" for n in algorithms_cfg
    )
    print(row)
