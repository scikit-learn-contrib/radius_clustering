"""
Radius Clustering

This module provides functionality for Minimum Dominating Set (MDS) based clustering.
It includes methods for solving MDS problems and applying the solutions to
clustering tasks.

This module serves as the main interface for the Radius clustering library.
"""

import os
import numpy as np
import scipy.spatial as sp_spatial
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array

from radius_clustering.utils._emos import py_emos_main
from radius_clustering.utils._mds_approx import solve_mds

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class RadiusClustering(BaseEstimator, ClusterMixin):
    """
    Radius Clustering algorithm.

    This class implements clustering based on the Minimum Dominating Set (MDS) problem.
    It can use either an exact or approximate method for solving the MDS problem.

    Parameters:
    -----------
    manner : str, optional (default="approx")
        The method to use for solving the MDS problem. Can be "exact" or "approx".
    threshold : float, optional (default=0.5)
        The dissimilarity threshold to act as radius constraint for the clustering.

    Attributes:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data.
    centers : list
        The indices of the cluster centers.
    labels\_ : array-like, shape (n_samples,)
        The cluster labels for each point in the input data.
    effective_radius : float
        The maximum distance between any point and its assigned cluster center.
    """

    def __init__(self, manner="approx", threshold=0.5):
        self.manner = manner
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit the MDS clustering model to the input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.
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
        self.X = check_array(X)

        # Create dist and adj matrices
        dist_mat = sp_spatial.distance_matrix(self.X, self.X)
        adj_mask = np.triu((dist_mat <= self.threshold), k=1)
        self.nb_edges = np.sum(adj_mask)
        self.edges = np.argwhere(adj_mask).astype(np.int32)
        self.dist_mat = dist_mat

        self._clustering()
        self._compute_effective_radius()
        self._compute_labels()

        return self

    def fit_predict(self, X, y=None):
        """
        Fit the model and return the cluster labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.
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
        n = self.X.shape[0]
        if self.manner == "exact":
            self._clustering_exact(n)
        else:
            self._clustering_approx(n)

    def _clustering_exact(self, n):
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
        self.centers_, self._mds_exec_time = py_emos_main(
            self.edges.flatten(), n, self.nb_edges
        )

    def _clustering_approx(self, n):
        """
        Perform approximate MDS clustering.

        Parameters:
        -----------
        n : int
            The number of points in the dataset.

        Notes:
        ------
        This function uses the approximation method to solve the MDS problem.
        See [casado]_ for more details.
        """
        result = solve_mds(n, self.edges.flatten(), self.nb_edges, "test")
        self.centers_ = [x for x in result["solution_set"]]
        self._mds_exec_time = result["Time"]

    def _compute_effective_radius(self):
        """
        Compute the effective radius of the clustering.

        The effective radius is the maximum radius among all clusters.
        That means EffRad = max(R(C_i)) for all i.
        """
        self.effective_radius = np.min(self.dist_mat[:, self.centers_], axis=1).max()

    def _compute_labels(self):
        """
        Compute the cluster labels for each point in the dataset.
        """
        distances = self.dist_mat[:, self.centers_]
        self.labels_ = np.argmin(distances, axis=1)

        min_dist = np.min(distances, axis=1)
        self.labels_[min_dist > self.threshold] = -1
