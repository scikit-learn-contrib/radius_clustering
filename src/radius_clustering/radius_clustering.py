"""
Radius Clustering

This module provides functionality for Minimum Dominating Set (MDS) based clustering.
It includes methods for solving MDS problems and applying the solutions to
clustering tasks.

This module serves as the main interface for the Radius clustering library.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_random_state, validate_data

from radius_clustering.utils._emos import py_emos_main
from radius_clustering.utils._mds_approx import solve_mds

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class RadiusClustering(ClusterMixin, BaseEstimator):
    r"""
    Radius Clustering algorithm.

    This class implements clustering based on the Minimum Dominating Set (MDS) problem.
    It can use either an exact or approximate method for solving the MDS problem.

    Parameters:
    -----------
    manner : str, optional (default="approx")
        The method to use for solving the MDS problem. Can be "exact" or "approx".
    radius : float, optional (default=0.5)
        The dissimilarity threshold to act as radius constraint for the clustering.

    Attributes:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data.
    centers\_ : list
        The indices of the cluster centers.
    labels\_ : array-like, shape (n_samples,)
        The cluster labels for each point in the input data.
    effective_radius\_ : float
        The maximum distance between any point and its assigned cluster center.
    random_state\_ : int | None
        The random state used for reproducibility. If None, no random state is set.

    .. note::
        The `random_state_` attribute is not used when the `manner` is set to "exact".

    .. versionadded:: 1.3.0
        The *random_state* parameter was added to allow reproducibility in
        the approximate method.

    .. versionchanged:: 1.3.0
        All publicly accessible attributes are now suffixed with an underscore
        (e.g., `centers_`, `labels_`).
        This is particularly useful for compatibility with scikit-learn's API.

    .. versionadded:: 1.3.0
        The `radius` parameter replaces the `threshold` parameter for setting
        the dissimilarity threshold for better clarity and consistency.

    .. deprecated:: 1.3.0
        The `threshold` parameter is deprecated. Use `radius` instead.
        Will be removed in a future version.
    """

    _estimator_type = "clusterer"

    def __init__(
        self,
        manner: str = "approx",
        radius: float = 0.5,
        threshold=None,
        random_state: int | None = None,
    ) -> None:
        if threshold is not None:
            warnings.warn(
                "The 'threshold' parameter is deprecated and"
                " will be removed in a future version."
                "Please use 'radius' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            radius = threshold
        self.threshold = threshold  # For backward compatibility
        self.manner = manner
        self.radius = radius
        self.random_state = random_state

    def _check_symmetric(self, a: np.ndarray, tol: float = 1e-8) -> bool:
        if a.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        if a.shape[0] != a.shape[1]:
            return False
        return np.allclose(a, a.T, atol=tol)

    def fit(self, X: np.ndarray, y: None = None) -> "RadiusClustering":
        """
        Fit the MDS clustering model to the input data.

        This method computes the distance matrix if the input is a feature matrix,
        or uses the provided distance matrix directly if the input is already
        a distance matrix.

        .. note::
            If the input is a distance matrix, it should be symmetric and square.
            If the input is a feature matrix, the distance matrix
            will be computed using Euclidean distance.

        .. tip::
            Next version will support providing different metrics or
            even custom callables to compute the distance matrix.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster. X should be a 2D array-like structure.
            It can either be :
            - A distance matrix (symmetric, square) with shape (n_samples, n_samples).
            - A feature matrix with shape (n_samples, n_features)
            where the distance matrix will be computed.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        --------
        self : object
            Returns self.

        Examples :
        ----------

        >>> from radius_clustering import RadiusClustering
        >>> from sklearn import datasets
        >>> # Load the Iris dataset
        >>> iris = datasets.fetch_openml(name="iris", version=1, parser="auto")
        >>> X = iris["data"]  # Use dictionary-style access instead of attribute access
        >>> rad = RadiusClustering(manner="exact", threshold=1.43).fit(
        ...     X
        ... )  # Threshold set to 1.43 because it is the optimal
        ... # threshold for the Iris dataset
        >>> rad.centers_
        [96, 49, 102]

        For examples on common datasets and differences with kmeans,
        see :ref:`sphx_glr_auto_examples_plot_iris_example.py`
        """
        self.X_checked_ = validate_data(self, X)

        # Create dist and adj matrices
        if not self._check_symmetric(self.X_checked_):
            dist_mat = pairwise_distances(self.X_checked_, metric="euclidean")
        else:
            dist_mat = self.X_checked_

        if not isinstance(self.radius, (float, int)):
            raise ValueError("Radius must be a positive float.")
        if self.radius <= 0:
            raise ValueError("Radius must be a positive float.")
        adj_mask = np.triu((dist_mat <= self.radius), k=1)
        self.nb_edges_ = np.sum(adj_mask)
        if self.nb_edges_ == 0:
            self.centers_ = list(range(self.X_checked_.shape[0]))
            self.labels_ = np.array(self.centers_)
            self.effective_radius_ = 0
            self.mds_exec_time_ = 0
            return self
        self.edges_ = np.argwhere(adj_mask).astype(
            np.uint32
        )  # Edges in the adjacency matrix
        # uint32 is used to use less memory. Max number of features is 2^32-1
        self.dist_mat_ = dist_mat

        self._clustering()
        self._compute_effective_radius()
        self._compute_labels()

        return self

    def fit_predict(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Fit the model and return the cluster labels.

        This method is a convenience function that combines `fit` and `predict`.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster. X should be a 2D array-like structure.
            It can either be :
            - A distance matrix (symmetric, square) with shape (n_samples, n_samples).
            - A feature matrix with shape (n_samples, n_features) where
            the distance matrix will be computed.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns:
        --------
        labels : array, shape (n_samples,)
            The cluster labels for each point in X.
        """
        self.fit(X)
        return self.labels_

    def _clustering(self):
        """
        Perform the clustering using either the exact or approximate MDS method.
        """
        n = self.X_checked_.shape[0]
        if self.manner != "exact" and self.manner != "approx":
            print(f"Invalid manner: {self.manner}. Defaulting to 'approx'.")
            raise ValueError("Invalid manner. Choose either 'exact' or 'approx'.")
        if self.manner == "exact":
            self._clustering_exact(n)
        else:
            self._clustering_approx(n)

    def _clustering_exact(self, n: int) -> None:
        """
        Perform exact MDS clustering.

        Parameters:
        -----------
        n : int
            The number of points in the dataset.

        Notes:
        ------
        This function uses the EMOS algorithm to solve the MDS problem.
        See: [jiang]_ for more details.
        """
        self.centers_, self.mds_exec_time_ = py_emos_main(
            self.edges_.flatten(), n, self.nb_edges_
        )
        self.centers_.sort()  # Sort the centers to ensure consistent order

    def _clustering_approx(self, n: int) -> None:
        """
        Perform approximate MDS clustering.
        This method uses a pretty trick to set the seed for
        the random state of the C++ code of the MDS solver.

        .. tip::
            The random state is used to ensure reproducibility of the results
            when using the approximate method.
            If `random_state` is None, a default value of 42 is used.

        .. important::
            :collapsible: closed
            The trick to set the random state is :
            1. Use the `check_random_state` function to get a `RandomState`singleton
            instance, set up with the provided `random_state`.
            2. Use the `randint` method of the `RandomState` instance to generate a
            random integer.
            3. Use this random integer as the seed for the C++ code of the MDS solver.

            This ensures that the seed passed to the C++ code is always an integer,
            which is required by the MDS solver, and allows for
            reproducibility of the results.

        Parameters:
        -----------
        n : int
            The number of points in the dataset.

        Notes:
        ------
        This function uses the approximation method to solve the MDS problem.
        See [casado]_ for more details.
        """
        if self.random_state is None:
            self.random_state = 42
        self.random_state_ = check_random_state(self.random_state)
        seed = self.random_state_.randint(np.iinfo(np.int32).max)
        result = solve_mds(
            n, self.edges_.flatten().astype(np.int32), self.nb_edges_, seed
        )
        self.centers_ = sorted([x for x in result["solution_set"]])
        self.mds_exec_time_ = result["Time"]

    def _compute_effective_radius(self):
        """
        Compute the effective radius of the clustering.

        The effective radius is the maximum radius among all clusters.
        That means EffRad = max(R(C_i)) for all i.
        """
        self.effective_radius_ = np.min(self.dist_mat_[:, self.centers_], axis=1).max()

    def _compute_labels(self):
        """
        Compute the cluster labels for each point in the dataset.
        """
        distances = self.dist_mat_[:, self.centers_]
        self.labels_ = np.argmin(distances, axis=1)

        min_dist = np.min(distances, axis=1)
        self.labels_[min_dist > self.radius] = -1
