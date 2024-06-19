cimport cython
import numpy as np
cimport numpy as np
from scipy.special.cython_special cimport entr

np.import_array()

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float get_entropy(float[::1] proba_vec):
    """Return the Shannon entropy of a probability vector"""
    cdef:
        Py_ssize_t k
        Py_ssize_t n = proba_vec.shape[0]
        float out = 0.0

    for k from 0 <= k < n:
        out += entr(proba_vec[k])
    return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float[:, ::1] get_cumsum(float[:, ::1] signal):
    """Return a array of cumulative sums.
    
    if signal is of shape (n_samples, n_dims), the output is of shape
    (n_samples+1, n_dims). The first element is zero.
    """
    cdef:
        Py_ssize_t n_samples = signal.shape[0]
        Py_ssize_t n_dims = signal.shape[1]
        Py_ssize_t k_sample, k_dim
        float[::1] running_sum_vec = np.zeros(n_dims, dtype=np.float32, order="C")
        float[:, ::1] out = np.empty((n_samples+1, n_dims), dtype=np.float32, order="C")

    out[0, :] = 0
    for k_sample from 0 < k_sample <= n_samples:
        for k_dim from 0 <= k_dim < n_dims:
            running_sum_vec[k_dim] += signal[k_sample-1, k_dim]
            out[k_sample, k_dim] = running_sum_vec[k_dim]
    return out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef float[:, ::1] get_cumsum_sq(float[:, ::1] signal):
    """Return an array of cumulative sums of squared elements.
    
    if signal is of shape (n_samples, n_dims), the output is of shape
    (n_samples+1, n_dims). The first element is zero.
    """
    cdef:
        Py_ssize_t n_samples = signal.shape[0]
        Py_ssize_t n_dims = signal.shape[1]
        Py_ssize_t k_sample, k_dim
        float[::1] running_sum_vec = np.zeros(n_dims, dtype=np.float32, order="C")
        float[:, ::1] out = np.empty((n_samples+1, n_dims), dtype=np.float32, order="C")

    out[0, :] = 0
    for k_sample from 0 < k_sample <= n_samples:
        for k_dim from 0 <= k_dim < n_dims:
            running_sum_vec[k_dim] += signal[k_sample-1, k_dim]**2
            out[k_sample, k_dim] = running_sum_vec[k_dim]
    return out
