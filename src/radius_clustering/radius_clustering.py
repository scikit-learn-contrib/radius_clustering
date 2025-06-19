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

from .algorithms import clustering_approx, clustering_exact

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
    
    .. versionchanged:: 1.4.0
        The `RadiusClustering` class has been refactored.
        Clustering algorithms are now separated into their own module
        (`algorithms.py`) to improve maintainability and extensibility.
    
    .. versionadded:: 1.4.0
        The `set_solver` method was added to allow users to set a custom solver
        for the MDS problem. This allows for flexibility in how the MDS problem is solved
        and enables users to use their own implementations of MDS clustering algorithms.

    .. versionadded:: 1.3.0

        - The *random_state* parameter was added to allow reproducibility in the approximate method.

        - The `radius` parameter replaces the `threshold` parameter for setting the dissimilarity threshold for better clarity and consistency.

    .. versionchanged:: 1.3.0
        All publicly accessible attributes are now suffixed with an underscore
        (e.g., `centers_`, `labels_`).
        This is particularly useful for compatibility with scikit-learn's API.

    .. deprecated:: 1.3.0
        The `threshold` parameter is deprecated. Use `radius` instead.
        Will be removed in a future version.
    """

    _estimator_type = "clusterer"
    _algorithms = {
        "exact": clustering_exact,
        "approx": clustering_approx,
    }

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

    def fit(self, X: np.ndarray, y: None = None, metric: str | callable = "euclidean") -> "RadiusClustering":
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

        metric : str | callable, optional (default="euclidean")
            The metric to use when computing the distance matrix.
            The default is "euclidean".
            This should be a valid metric string from
            `sklearn.metrics.pairwise_distances` or a callable that computes
            the distance between two points.
        
        .. note::
            The metric parameter *MUST* be a valid metric string from
            `sklearn.metrics.pairwise_distances` or a callable that computes
            the distance between two points.
            Valid metric strings include :
            - "euclidean"
            - "manhattan"
            - "cosine"
            - "minkowski"
            - and many more supported by scikit-learn.
            please refer to the
            `sklearn.metrics.pairwise_distances` documentation for a full list.
        
        .. attention::
            If the input is a distance matrix, the metric parameter is ignored.
            The distance matrix should be symmetric and square.
        
        .. warning::
            If the parameter is a callable, it should :
            - Accept two 1D arrays as input.
            - Return a single float value representing the distance between the two points.

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
            dist_mat = pairwise_distances(self.X_checked_, metric=metric)
        else:
            dist_mat = self.X_checked_
        
        if not self._check_symmetric(dist_mat):
            raise ValueError("Input distance matrix must be symmetric. Got a non-symmetric matrix.")
        self.dist_mat_ = dist_mat
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
        self.clusterer_ = self._algorithms.get(self.manner, self._algorithms["approx"])
        self._clustering()
        self._compute_effective_radius()
        self._compute_labels()

        return self

    def fit_predict(self, X: np.ndarray, y: None = None, metric: str | callable = "euclidean") -> np.ndarray:
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
        
        metric : str | callable, optional (default="euclidean")
            The metric to use when computing the distance matrix.
            The default is "euclidean".
            Refer to the `fit` method for more details on valid metrics.

        Returns:
        --------
        labels : array, shape (n_samples,)
            The cluster labels for each point in X.
        """
        self.fit(X, metric=metric)
        return self.labels_

    def _clustering(self):
        """
        Perform the clustering using either the exact or approximate MDS method.
        """
        n = self.X_checked_.shape[0]
        if self.manner not in self._algorithms:
            raise ValueError(f"Invalid manner. Please choose in {list(self._algorithms.keys())}.")
        if self.clusterer_ == clustering_approx:
            if self.random_state is None:
                self.random_state = 42
            self.random_state_ = check_random_state(self.random_state)
            seed = self.random_state_.randint(np.iinfo(np.int32).max)
        else:
            seed = None
        self.centers_, self.mds_exec_time_ = self.clusterer_(n, self.edges_, self.nb_edges_, seed)

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

    def set_solver(self, solver: callable) -> None:
        """
        Set a custom solver for resolving the MDS problem.
        This method allows users to replace the default MDS solver with a custom one.

        An example is provided below and in the example gallery : 
        :ref:`sphx_glr_auto_examples_plot_benchmark_custom.py`

        .. important::
            The custom solver must accept the same parameters as the default solvers
            and return a tuple containing the cluster centers and the execution time.
            e.g., it should have the signature:
            
            >>> def custom_solver(
            >>>             n: int,
            >>>             edges: np.ndarray,
            >>>             nb_edges: int,
            >>>             random_state: int | None = None
            >>>         ) -> tuple[list, float]:
            >>>     # Custom implementation details
            >>>     centers = [...]
            >>>     exec_time = ...
            >>>     # Return the centers and execution time
            >>>     return centers, exec_time
            
            This allows for flexibility in how the MDS problem is solved.

        Parameters:
        -----------
        solver : callable
            The custom solver function to use for MDS clustering.
            It should accept the same parameters as the default solvers
            and return a tuple containing the cluster centers and the execution time.

        Raises:
        -------
        ValueError
            If the provided solver does not have the correct signature.

        """
        if not callable(solver):
            raise ValueError("The provided solver must be callable.")
        
        # Check if the solver has the correct signature
        try:
            n = 3
            edges = np.array([[0, 1], [1, 2], [2, 0]])
            nb_edges = edges.shape[0]
            solver(n, edges, nb_edges, random_state=None)
        except Exception as e:
            raise ValueError(f"The provided solver does not have the correct signature: {e}") from e
        self.manner = "custom"
        self._algorithms["custom"] = solver