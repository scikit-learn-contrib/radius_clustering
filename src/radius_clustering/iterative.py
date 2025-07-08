"""
This module contains the implementation of the CURGRAPH algorithm.

The CURGRAPH algorithm is an iterative algorithm that takes as input:

    * a set of data and features
    * a number of max clusters to find (optional)
    * a threshold for the radius constraint (optional)

The algorithm returns whether:

    * the clusters and the maximum radius optimally found for nb_cluster from 2 to max clusters
    * the clusters and the maximum radius optimally found for a given threshold
    * the clusters and the maximum radius optimally found for a given number of clusters

The algorithm is based on the following steps:

    1. Compute the distance matrix between the data
    2. Rank the dissimilarities in decreasing order
    3. If max number of clusters is given:
        For each dissimilarity until max number of clusters is reached:
            1. Compute the corresponding input graph considering each
            dissimilarity above the dissimilarity threshold as no edge
            2. Find the minimum dominating set of the graph
            3. If the cardinality of the MDS is above the previous
            cardinality found, store the threshold,the MDS and the max radius
            4. If not, continue to the next dissimilarity
        returns the cardinality of the MDS, the threshold associated
        and the max radius
    4. If threshold is given:
        1. Compute the corresponding input graph considering each
        dissimilarity above the dissimilarity threshold as no edge
        2. Find the minimum dominating set of the graph
        3. Compute max radius and cardinality of the MDS
        For each dissimilarity until cardinality of the MDS is above the previous cardinality found:
            1. Compute the corresponding input graph considering each
            dissimilarity above the dissimilarity threshold as no edge
            2. Find the minimum dominating set of the graph
            3. If the cardinality of the MDS is above the previous
            cardinality found, store the previous threshold, MDS and max radius
            4. If not, continue to the next dissimilarity
        returns the cardinality of the MDS, the MDS, the max radius and the threshold associated
"""

from __future__ import annotations

import os
import time
import numpy as np
import scipy as sp

from copy import deepcopy
from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from typing import Union, List, Dict, Any, Tuple

from .radius_clustering import RadiusClustering

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.getcwd()


def gen_even_slices(n, n_packs, n_samples=None):
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


class Curgraph(object):
    """
    CURGRAPH algorithm for clustering based on Minimum Dominating Set (MDS).

    The CURGRAPH algorithm is an iterative algorithm that have been designed upon
    CLUSTERGRAPH algorithm, presented by Hansen and Delattre (1978)
    in "Complete-link cluster analysis by graph coloring".

    Parameters:
    ----------
    max_clusters : int, optional
        The maximum number of clusters to consider. Default is None.
    radius : Union[int, float], optional
        The dissimilarity threshold for clustering. Default is None.
    manner : {'approx', 'exact'}, optional
        The manner in which to compute the clusters. 'approx' uses an approximate method,
        while 'exact' uses an exact method. Default is 'approx'.
    random_state : int, RandomState instance or None, optional
        Controls the randomness of the estimator. Pass an int for reproducible output across multiple function calls.

    Attributes:
    ----------
    results_ : dict
        A dictionary to store the results of the CURGRAPH algorithm.

    Methods:
    -------
    fit(X: ArrayLike, y: None) -> self:
        Run the CURGRAPH algorithm.
    predict(X: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        Predict the clusters and maximum radius for the given data, for a specific number of clusters.
    get_results() -> dict:
        Return the results of the CURGRAPH algorithm.
    """

    def __init__(
        self,
        manner: str = "approx",
        max_clusters: int = None,
        radius: Union[int, float] = None,
    ):
        self.manner = manner
        self.max_clusters = max_clusters
        self.radius = radius
        self.results_ = {}
        self.solver = RadiusClustering(manner=self.manner)

    def _init_dist_list(self, X: np.ndarray) -> None:
        """
        Initialize the list of dissimilarities based on the radius parameter.
        """
        self.X_ = X
        self.dist_mat_ = pairwise_distances(self.X_)
        self._list_t = np.unique(self.dist_mat_)[::-1]
        if self.radius is None:
            t = self.dist_mat_.max(axis=1).min()
        else:
            t = self.radius
        radius = t
        arg_radius = np.where(self._list_t <= radius)[0]
        self._list_t = self._list_t[arg_radius:]

    def _init_results(self):
        if self.max_clusters is not None:
            for i in range(2, self.max_clusters + 1):
                self.results_[i] = {"radius": None, "centers": None}

    def fit(self, X: ArrayLike, y: None, n_jobs: int = -1) -> self:
        """
        Run the CURGRAPH algorithm.
        """
        self._init_results()
        self._init_dist_list(X)
        self.n_jobs_ = effective_n_jobs(n_jobs)
        dissimilarity_index = 0
        first_t = self._list_t[0]
        old_mds = self.solver.set_params(radius=first_t).fit(X).centers_
        cardinality_limit = (
            self.max_clusters + 1 if self.max_clusters else len(old_mds) + 1
        )
        tasks = [
            delayed(self._curgraph)(
                dissimilarity_index, self._list_t[s], old_mds, cardinality_limit
            )
            for s in gen_even_slices(len(self._list_t), self.n_jobs_)
        ]
        with parallel_backend("threading", n_jobs=self.n_jobs_):
            Parallel()(tasks)

        return self

    def _curgraph(
        self,
        index_d: int,
        list_t: List[float],
        old_mds: np.ndarray,
        cardinality_limit: int,
    ) -> None:
        while (len(old_mds) < cardinality_limit) and (index_d < len(list_t)):
            old_mds = self._process_mds(index_d, list_t, old_mds)
            index_d += 1

    def _process_mds(
        self, index_d: int, list_t: List[float], old_mds: np.ndarray
    ) -> np.ndarray:
        """
        Process the minimum dominating set (MDS) for a given dissimilarity index.
        """
        t = list_t[index_d]
        if self._is_dominating_set(t, old_mds):
            return old_mds
        new_mds = self.solver.set_params(radius=t).fit(self.X_).centers_
        if len(new_mds) > len(old_mds):
            self._update_results(t, new_mds)

        return new_mds

    def _update_results(self, mds: np.ndarray, t: float) -> None:
        """
        Update the results dictionary with the new MDS and radius.
        """
        card = len(mds)
        if self.results_[card]:
            if t < self.results_[card]["radius"]:
                self.results_[card] = {"radius": t, "centers": mds}
        else:
            self.results_[card] = {"radius": t, "centers": mds}

    def _is_dominating_set(self, t: float, mds: np.ndarray) -> bool:
        """
        Check if the current MDS is a dominating set for the given threshold t.
        """
        adj_mat = self.dist_mat_ <= t
        return np.all(np.any(adj_mat[:, mds], axis=1))

    def predict(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        """
        Predict the clusters and maximum radius for the given data, for a specific number of clusters.
        """
        # Implementation of the predict method
        return np.array([]), 0.0

    def get_results(self) -> dict:
        """
        Return the results of the CURGRAPH algorithm.
        """
        return self.results_
