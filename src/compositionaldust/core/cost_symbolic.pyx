cimport cython
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from scipy.special.cython_special cimport entr

from compositionaldust.core.utils cimport get_cumsum, get_entropy

np.import_array()


cdef class CostSymbolic:

    def __cinit__(self, float[:, ::1] signal):
        self.signal = signal
        self.signal_cumsum = get_cumsum(signal)
        self.n_samples = signal.shape[0]
        self.n_dims = signal.shape[1]
        self.prob_vec_tmp = np.empty((self.n_dims,), dtype=np.float32, order="C")


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef float evaluate(self, Py_ssize_t start, Py_ssize_t end):
        cdef:
            float res = 0.0
            float empirical_prob
            float length = end - start
            Py_ssize_t k_dim

        for k_dim from 0 <= k_dim < self.n_dims:
            empirical_prob = (self.signal_cumsum[end, k_dim]-self.signal_cumsum[start, k_dim]) / length
            res += length * entr(empirical_prob)
        return res


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef float compute_entropy(self, Py_ssize_t start, Py_ssize_t mid, Py_ssize_t end, float mu):
        cdef:
            float scale_factor = (end - mid) - mu * (mid - start)
            float empirical_prob
            Py_ssize_t k_dim

        for k_dim from 0 <= k_dim < self.n_dims:
            empirical_prob = (self.signal_cumsum[end, k_dim] - self.signal_cumsum[mid, k_dim])
            empirical_prob -= mu * (self.signal_cumsum[mid, k_dim] - self.signal_cumsum[start, k_dim])
            empirical_prob /= scale_factor
            self.prob_vec_tmp[k_dim] = empirical_prob

        return get_entropy(self.prob_vec_tmp)
        

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.nonecheck(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef float compute_mu_max(self, Py_ssize_t start, Py_ssize_t mid, Py_ssize_t end):
        cdef:
            float mu_max, right_count, left_count
            Py_ssize_t k_dim

        mu_max = INFINITY
        for k_dim from 0 <= k_dim < self.n_dims:
            left_count = self.signal_cumsum[mid, k_dim] - self.signal_cumsum[start, k_dim]
            right_count = self.signal_cumsum[end, k_dim] - self.signal_cumsum[mid, k_dim]
            if left_count > 0:
                if right_count > 0:
                    mu_max = min(mu_max, right_count / left_count)
                else:
                    return 0.0
        return mu_max
