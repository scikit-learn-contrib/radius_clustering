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
import random
import warnings
import numpy as np
import scipy as sp

from copy import deepcopy
from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances
from typing import Union, List, Dict, Any, Tuple, Optional

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


class Curgraph(ClusterMixin, BaseEstimator):
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

    _estimator_type = "clusterer"

    def __init__(
        self,
        manner: str = "approx",
        max_clusters: int = None,
        n_clusters: Union[int, None] = None,
        random_state: Union[int, np.random.RandomState, None] = None,
        n_jobs: int = -1,
    ):
        self.manner = manner
        self.max_clusters = max_clusters
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _check_symmetric(self, X: np.ndarray) -> bool:
        """
        Check if the input matrix is symmetric.
        """
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            return False
        return np.allclose(X, X.T, atol=1e-8)

    def _init_dist_list(self, radius: Optional[int | float]) -> float:
        """
        Initialize the list of dissimilarities based on the radius parameter.
        """
        if not self._check_symmetric(self.X_):
            self.dist_mat_ = pairwise_distances(self.X_)
        else:
            self.dist_mat_ = self.X_
        tril_mat = np.tril(self.dist_mat_, k=-1)
        self._list_t = np.unique(tril_mat)[::-1][:-1]  # Exclude the zero distance
        if radius is None:
            t = self.dist_mat_.max(axis=1).min()
        else:
            if not isinstance(radius, (int, float)):
                raise ValueError(
                    f"Radius must be an int or float, got {type(radius)} instead."
                )
            if radius <= 0:
                warnings.warn(
                    f"Radius must be a positive float, got {radius}.\n"
                    "Defaulting radius to the MinMax in distance matrix.\n"
                    "See documentation for more details.",
                    UserWarning,
                    stacklevel=2,
                )
                radius = self.dist_mat_.max(axis=1).min()
            t = radius
        arg_radius = np.where(self._list_t <= t)[0][0]
        self._list_t = self._list_t[arg_radius:]
        return t

    def fit(
        self, X: np.ndarray, y=None, radius: Optional[int | float] = None
    ) -> "Curgraph":
        """
        Run the CURGRAPH algorithm.
        """
        self.results_ = {}
        self.X_ = validate_data(self, X, ensure_all_finite=True)
        first_t = self._init_dist_list(radius)
        self.solver_ = RadiusClustering(
            manner=self.manner, random_state=self.random_state
        )
        if radius is not None:
            dissimilarity_index = 0
            first_t = self._list_t[dissimilarity_index]
            old_mds = self.solver_.set_params(radius=first_t).fit(X).centers_
        else:
            dissimilarity_index = 1
            old_mds = [np.argmin(self.dist_mat_.max(axis=1))]
            self.results_[1] = {"radius": first_t, "centers": old_mds}
        cardinality_limit = (
            self.max_clusters + 1 if self.max_clusters else len(old_mds) + 1
        )
        tasks = [
            delayed(Curgraph._curgraph)(
                dissimilarity_index,
                self._list_t[s],
                old_mds,
                cardinality_limit,
                self.dist_mat_,
                self.solver_,
            )
            for s in gen_even_slices(len(self._list_t), effective_n_jobs(self.n_jobs))
        ]
        with parallel_backend("threading", n_jobs=effective_n_jobs(self.n_jobs)):
            result_list = Parallel()(tasks)

        for local_results in result_list:
            for card, result in local_results.items():
                if card not in self.results_:
                    self.results_[card] = result
                else:
                    if result["radius"] < self.results_[card]["radius"]:
                        self.results_[card] = result
        if self.n_clusters is None:
            n_clusters = random.choice(list(self.results_.keys()))
        else:
            n_clusters = self.n_clusters
        target_centers = self.results_.get(n_clusters, {}).get("centers", [])
        try:
            self.labels_ = self._compute_labels(target_centers)
        except Exception as e:
            if self.results_:
                warnings.warn(
                    f"An error occurred while computing labels: {e}\n"
                    "Defaulting to n_cluster=1 for algorithm continuity\n"
                    f"NB clusters available for pickup : {sorted(self.results_.keys())}",
                    UserWarning,
                    stacklevel=2,
                )
                self.labels_ = np.zeros(self.X_.shape[0], dtype=int)
            else:
                raise ValueError(
                    "No clusters found. Please check the input data and parameters."
                ) from e
        return self

    def predict_new_data(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = validate_data(self, X, ensure_all_finite=True, reset=False)
        solution = self.results_.get(self.n_clusters, None)
        if solution is None:
            n_to_predict = random.choice(self.available_clusters)
            warnings.warn(
                f"No solution found for n_clusters={n_to_predict}.\n"
                f"Available clusters: {self.available_clusters}\n"
                f"Defaulting to {n_to_predict} clusters.\n",
                UserWarning,
                stacklevel=2,
            )
            solution = self.results_.get(n_to_predict, None)
        centers = solution["centers"]
        distance_to_centers = self.dist_mat_[:, centers]
        return np.argmin(distance_to_centers, axis=1)

    @property
    def available_clusters(self) -> list[int]:
        check_is_fitted(self)
        return sorted(self.results_.keys())

    @staticmethod
    def _curgraph(
        index_d: int,
        list_t: List[float],
        old_mds: np.ndarray,
        cardinality_limit: int,
        dist_mat: np.ndarray,
        solver: RadiusClustering,
    ) -> Dict[int, Dict[str, Any]]:
        local_results = {}
        while (len(old_mds) < cardinality_limit) and (index_d < len(list_t)):
            old_mds = Curgraph._process_mds(
                index_d, list_t, old_mds, local_results, solver, dist_mat
            )
            index_d += 1

        if len(old_mds) <= cardinality_limit:
            # If the MDS is smaller than the limit, we store it
            # with the radius corresponding to the last dissimilarity index.

            local_results[len(old_mds)] = {
                "radius": list_t[index_d - 1],
                "centers": old_mds,
            }
        return local_results

    @staticmethod
    def _process_mds(
        index_d: int,
        list_t: List[float],
        old_mds: np.ndarray,
        local_results: Dict[int, Dict[str, Any]],
        solver: RadiusClustering,
        dist_mat: np.ndarray,
    ) -> np.ndarray:
        """
        Process the minimum dominating set (MDS) for a given dissimilarity index.
        """
        t = list_t[index_d]
        print(f"THRESHOLD : {t}")
        if Curgraph._is_dominating_set(t, old_mds, dist_mat):
            return old_mds
        new_mds = solver.set_params(radius=t).fit(dist_mat).centers_
        if len(new_mds) > len(old_mds):
            Curgraph._update_results(mds=new_mds, t=t, local_results=local_results)

        return new_mds

    @staticmethod
    def _update_results(
        mds: np.ndarray, t: float, local_results: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Update the results dictionary with the new MDS and radius.
        """
        card = len(mds)
        if card not in local_results:
            local_results[card] = {"radius": t, "centers": mds}
        else:
            if t < local_results[card]["radius"]:
                local_results[card] = {"radius": t, "centers": mds}
            else:
                local_results[card] = {"radius": t, "centers": mds}

    @staticmethod
    def _is_dominating_set(t: float, mds: np.ndarray, dist_mat: np.ndarray) -> bool:
        """
        Check if the current MDS is a dominating set for the given threshold t.
        """
        adj_mat = dist_mat <= t
        return np.all(np.any(adj_mat[:, mds], axis=1))

    def get_results(self) -> dict:
        """
        Return the results of the CURGRAPH algorithm.
        """
        return self.results_

    def _compute_labels(self, target_centers: np.ndarray) -> np.ndarray:
        """
        Compute the labels for the data points based on the target clusters.
        """
        distances_to_centers = self.dist_mat_[:, target_centers]
        partition_radius = self.results_.get(self.n_clusters, {}).get("radius")
        labels = np.argmin(distances_to_centers, axis=1)
        min_distances = np.min(distances_to_centers, axis=1)
        labels[min_distances > partition_radius] = -1  # Assign -1 for outliers
        return labels
