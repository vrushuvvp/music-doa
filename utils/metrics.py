"""
Evaluation metrics for DOA estimation algorithms.
"""

import numpy as np


def rmse(estimated: np.ndarray, true_val: float) -> float:
    """
    Root Mean Square Error over Monte Carlo trials.

    Parameters
    ----------
    estimated : (n_trials,) array of estimated DOA values (degrees)
    true_val  : true DOA in degrees

    Returns
    -------
    rmse : float
    """
    return float(np.sqrt(np.mean((estimated - true_val) ** 2)))


def matched_rmse(estimated_batch: np.ndarray, true_vals: list) -> float:
    """
    RMSE for multiple sources: greedily matches estimated DOAs to true DOAs
    in each trial and averages.

    Parameters
    ----------
    estimated_batch : (n_trials, D) estimated DOAs per trial
    true_vals       : list of D true DOAs

    Returns
    -------
    mean_rmse : float
    """
    errors = []
    true_arr = np.array(true_vals)
    for est in estimated_batch:
        est_sorted = np.sort(est)
        true_sorted = np.sort(true_arr)
        # Pad/trim if detection count differs
        min_d = min(len(est_sorted), len(true_sorted))
        errors.append(np.mean((est_sorted[:min_d] - true_sorted[:min_d]) ** 2))
    return float(np.sqrt(np.mean(errors)))
