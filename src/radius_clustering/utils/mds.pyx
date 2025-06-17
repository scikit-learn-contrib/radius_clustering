"""
MDS Solver Module

This Cython module provides the core functionality for solving Minimum Dominating Set (MDS) problems.
It serves as a bridge between Python and the C++ implementation of the MDS algorithms.

The module includes:
- Wrapper functions for C++ MDS solvers
- Data structure conversions between Python/NumPy and C++
- Result processing and conversion back to Python objects
"""

# distutils: language = c++
# distutils: sources = mds_clustering/utils/mds_core.cpp mds_clustering/utils/random_manager.cpp

from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set as cpp_unordered_set
from libcpp.string cimport string
from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np

cdef extern from "random_manager.h":
    cdef cppclass RandomManager:
        @staticmethod
        void setSeed(long seed)

cdef extern from "mds_core.cpp":
    cdef cppclass Result:
        Result()
        Result(string instanceName)
        void add(string key, float value)
        float get(int pos)
        vector[string] getKeys()
        string getInstanceName()
        cpp_unordered_set[int] getSolutionSet()
        void setSolutionSet(cpp_unordered_set[int] solutionSet)

    cdef Result iterated_greedy_wrapper(int numNodes, const vector[int]& edges_list, int nb_edges, long seed) nogil

def solve_mds(int num_nodes, np.ndarray[int, ndim=1, mode="c"] edges not None, int nb_edges, int seed):
    """
    Solve the Minimum Dominating Set problem for a given graph.

    Parameters:
    -----------
    num_nodes : int
        The number of nodes in the graph.
    edges : np.ndarray
        A 1D NumPy array representing the edges of the graph.
    nb_edges : int
        The number of edges in the graph.
    name : str
        A name identifier for the problem instance.

    Returns:
    --------
    dict
        A dictionary containing the solution set and other relevant information.
    """
    cdef vector[int] cpp_edge_list
    
    # Cast the NumPy array to a C++ vector
    cpp_edge_list.assign(&edges[0], &edges[0] + edges.shape[0])
    
    cdef Result result
    with nogil:
        result = iterated_greedy_wrapper(num_nodes, cpp_edge_list, nb_edges, seed)

    # Convert the C++ Result to a Python dictionary
    py_result = {
        "solution_set": set(result.getSolutionSet()),
    }
    
    # Add other key-value pairs
    keys = result.getKeys()
    for i in range(len(keys)):
        py_result[keys[i].decode('utf-8')] = result.get(i)
    
    return py_result