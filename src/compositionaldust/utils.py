import numpy as np
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_signal(signal: np.ndarray):
    """One hot encode a symbolic signal"""
    err_msg = f"Signal must be 1D, not {signal.shape}."
    assert (signal.ndim == 1) or (signal.shape[1] == 1), err_msg
    ohe = OneHotEncoder(dtype=np.float32).fit(signal.reshape(-1, 1))
    transformed = np.asarray(ohe.transform(signal.reshape(-1, 1)).todense())
    return transformed


def from_path_matrix_to_bkps(path_vec):
    """Convert path matrix to list of change-point indexes."""
    path_vec = np.asarray(path_vec)
    n_samples = path_vec.shape[0] - 1
    bkps = list()
    bkp = n_samples
    while bkp > 0:
        bkps.append(bkp)
        bkp = int(path_vec[bkp])
    bkps.sort()
    return bkps
