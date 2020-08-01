# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import numpy as np


def normalize(arr):
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    arr_normalized = np.zeros_like(arr)
    if arr.ndim == 1:
        if not np.isclose(std, 0.0):
            arr_normalized = (arr - mean) / std
    else:
        idx = np.logical_not(np.isclose(std, 0.0))
        arr_normalized[:, idx] = (arr[:, idx] - mean[idx]) / std[idx]

    return arr_normalized


def compute_explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    # assert y.ndim == 1 and ypred.ndim == 1
    variance = np.zeros(y.shape[1])
    for idx in range(y.shape[1]):
        y_col = y[:, idx]
        ypred_col = ypred[:, idx]
        vary = np.var(y_col)
        # if np.isclose(vary, 0.0):
        #    if np.allclose(y_col, ypred_col):
        #        variance[idx] = 1.0
        #    else:
        #        variance[idx] = -1e5
        # else:
        variance[idx] = 1.0 - np.var(y_col - ypred_col) / vary
    return np.mean(variance)
