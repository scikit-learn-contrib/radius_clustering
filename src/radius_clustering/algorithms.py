"""
This module contains the implementation of the clustering algorithms.
It provides two main functions: `clustering_approx` and `clustering_exact`.

These functions can be replaced in the `RadiusClustering` class
to perform clustering using another algorithm.

.. versionadded:: 1.4.0
    Refactoring the structure of the code to separate the clustering algorithms
    This allows for easier maintenance and extensibility of the codebase.

"""
from __future__ import annotations

import numpy as np

from .utils._mds_approx import solve_mds
from .utils._emos import py_emos_main

def clustering_approx(
          n: int, edges: np.ndarray, nb_edges: int,
          random_state: int | None = None) -> None:
    """
    Perform approximate MDS clustering.
    This method uses a pretty trick to set the seed for
    the random state of the C++ code of the MDS solver.

    .. tip::
        The random state is used to ensure reproducibility of the results
        when using the approximate method.
        If `random_state` is None, a default value of 42 is used.

    .. important::
        The trick to set the random state is :

        1. Use the `check_random_state` function to get a `RandomState`singleton
        instance, set up with the provided `random_state`.

        2. Use the `randint` method of the `RandomState` instance to generate a
        random integer.

        3. Use this random integer as the seed for the C++ code of the MDS solver.

        
        This ensures that the seed passed to the C++ code is always an integer,
        which is required by the MDS solver, and allows for
        reproducibility of the results.
    
    .. note::
        This function uses the approximation method to solve the MDS problem.
        See [casado]_ for more details.

    Parameters:
    -----------
    n : int
        The number of points in the dataset.
    edges : np.ndarray
        The edges of the graph, flattened into a 1D array.
    nb_edges : int
        The number of edges in the graph.
    random_state : int | None
        The random state to use for reproducibility.
        If None, a default value of 42 is used.
    Returns:
    --------
    centers : list
        A sorted list of the centers of the clusters.
    mds_exec_time : float
        The execution time of the MDS algorithm in seconds.
    """
    result = solve_mds(
        n, edges.flatten().astype(np.int32), nb_edges, random_state
    )
    centers = sorted([x for x in result["solution_set"]])
    mds_exec_time = result["Time"]
    return centers, mds_exec_time

def clustering_exact(n: int, edges: np.ndarray, nb_edges: int, seed: None = None) -> None:
    """
    Perform exact MDS clustering.

    This function uses the EMOs algorithm to solve the MDS problem.

    .. important::
        The EMOS algorithm is an exact algorithm for solving the MDS problem.
        It is a branch and bound algorithm that uses graph theory tricks
        to efficiently cut the search space. See [jiang]_ for more details.

    Parameters:
    -----------
    n : int
        The number of points in the dataset.
    edges : np.ndarray
        The edges of the graph, flattened into a 1D array.
    nb_edges : int
        The number of edges in the graph.
    seed : None
        This parameter is not used in the exact method, but it is kept for
        compatibility with the approximate method.

    Returns:
    --------
    centers : list
        A sorted list of the centers of the clusters.
    mds_exec_time : float
        The execution time of the MDS algorithm in seconds.
    """
    centers, mds_exec_time = py_emos_main(
        edges.flatten(), n, nb_edges
    )
    centers.sort()
    return centers, mds_exec_time