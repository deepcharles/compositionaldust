cimport cython
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

cdef extern from "stdlib.h":
    void srand(unsigned int seed)

# Seed the random number generator
srand(time(NULL))

@cython.cdivision(True)
cdef float random_value():
    """Return a random float between 0 and 1."""
    cdef float rand_num = <float> rand()
    return rand_num / <float>RAND_MAX


@cython.cdivision(True)
cdef Py_ssize_t random_integer(Py_ssize_t lower, Py_ssize_t upper):
    """Return a random integer between lower (inclusive) and upper (inclusive)."""
    cdef:
        Py_ssize_t range_size = upper - lower + 1
        Py_ssize_t scaled_random = lower + rand() % range_size
    return scaled_random

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef Py_ssize_t choice_i(Py_ssize_t[::1] arr):
    """Return a random element of the array `arr`."""
    cdef Py_ssize_t size = arr.shape[0]
    return arr[random_integer(lower=0, upper=size-1)]
