cdef float get_entropy(float[::1] proba_vec)

cdef float[:, ::1] get_cumsum(float[:, ::1] signal)
cdef float[:, ::1] get_cumsum_sq(float[:, ::1] signal)