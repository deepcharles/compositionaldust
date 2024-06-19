cdef class CostSymbolic:
    cdef:
         float[:, ::1] signal
         float[::1] prob_vec_tmp
         float[:, ::1] signal_cumsum
         Py_ssize_t n_samples, n_dims

    cdef float evaluate(self, Py_ssize_t start, Py_ssize_t end)

    cdef float compute_entropy(self, Py_ssize_t start, Py_ssize_t mid, Py_ssize_t end, float mu)

    cdef float compute_mu_max(self, Py_ssize_t start, Py_ssize_t mid, Py_ssize_t end)