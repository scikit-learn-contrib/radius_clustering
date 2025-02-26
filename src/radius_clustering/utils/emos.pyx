"""
EMOS (Exact Minimum Dominating Set) Solver Module

This Cython module provides a Python interface to the C implementation
of the Exact Minimum Dominating Set (EMOS) algorithm. It allows for
efficient solving of MDS problems using the exact method.

The module includes:
- Wrapper functions for the C EMOS solver
- Data conversion between Python and C data structures
- Result processing and conversion back to Python objects
"""

cdef extern from "mds3-util.h":
    struct Result:
        int* dominating_set
        int set_size
        double exec_time

    Result* emos_main(unsigned int* edges, int nb_edge, int n)

    void cleanup()

    void free_results(Result* result)

import numpy as np
cimport numpy as np

def py_emos_main(np.ndarray[unsigned int, ndim=1] edges, int n, int nb_edge):
    cdef Result* result = emos_main(&edges[0], n, nb_edge)

    dominating_set = [result.dominating_set[i] - 1 for i in range(result.set_size)]
    exec_time = result.exec_time

    free_results(result)
    cleanup()

    return dominating_set, exec_time

