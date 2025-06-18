import pytest

from radius_clustering import RadiusClustering
from sklearn import datasets

X = datasets.fetch_openml(name="iris", version=1, parser="auto")["data"]

def test_radius_clustering_approx():
    """
    Test the approximate method of the RadiusClustering class.
    """
    clusterer = RadiusClustering(manner="approx", radius=1.43)

    assert clusterer.manner == "approx", "The manner should be 'approx'."
    assert clusterer.radius == 1.43, "The radius should be 1.43."
    assert clusterer.random_state is None, "The random state should be None by default."
    assert clusterer._estimator_type == "clusterer", "The estimator type should be 'clusterer'."
    assert clusterer._check_symmetric(X) is False, "The input should not be a symmetric distance matrix."

    clusterer.fit(X)

    assert clusterer.X_checked_ is not None, "X_checked_ should not be None after fitting."
    assert clusterer.dist_mat_ is not None, "dist_mat_ should not be None after fitting."
    assert clusterer.nb_edges_ > 0, "There should be edges in the graph."
    assert clusterer.labels_ is not None, "Labels should not be None after fitting."
    assert clusterer.centers_ is not None, "Centers should not be None after fitting."
    assert clusterer.effective_radius_ > 0, "Effective radius should be greater than 0."
    assert clusterer.mds_exec_time_ >= 0, "MDS execution time should be non-negative."
    assert clusterer.edges_ is not None, "Edges should not be None after fitting."
    assert clusterer.random_state == 42, "Random state should be set to 42 after fitting."

    results = clusterer.labels_
    assert len(results) == X.shape[0], "The number of labels should match the number of samples."
    assert len(set(results)) <= X.shape[0], "The number of unique labels should not exceed the number of samples."


def test_radius_clustering_exact():
    """
    Test the exact method of the RadiusClustering class.
    """
    clusterer = RadiusClustering(manner="exact", radius=1.43)

    assert clusterer.manner == "exact", "The manner should be 'exact'."
    assert clusterer.radius == 1.43, "The radius should be 1.43."
    assert clusterer.random_state is None, "The random state should be None by default."
    assert clusterer._estimator_type == "clusterer", "The estimator type should be 'clusterer'."
    assert clusterer._check_symmetric(X) is False, "The input should not be a symmetric distance matrix."

    clusterer.fit(X)

    assert clusterer.X_checked_ is not None, "X_checked_ should not be None after fitting."
    assert clusterer.dist_mat_ is not None, "dist_mat_ should not be None after fitting."
    assert clusterer.nb_edges_ > 0, "There should be edges in the graph."
    assert clusterer.labels_ is not None, "Labels should not be None after fitting."
    assert clusterer.centers_ is not None, "Centers should not be None after fitting."
    assert clusterer.effective_radius_ > 0, "Effective radius should be greater than 0."
    assert clusterer.mds_exec_time_ >= 0, "MDS execution time should be non-negative."
    assert clusterer.edges_ is not None, "Edges should not be None after fitting."
    assert clusterer.random_state is None, "Random state should remain None."

    results = clusterer.labels_
    assert len(results) == X.shape[0], "The number of labels should match the number of samples."
    assert len(set(results)) <= X.shape[0], "The number of unique labels should not exceed the number of samples."

def test_radius_clustering_fit_predict():
    """
    Test the fit_predict method of the RadiusClustering class.
    """
    clusterer = RadiusClustering(manner="approx", radius=1.43)

    assert clusterer.manner == "approx", "The manner should be 'approx'."
    assert clusterer.radius == 1.43, "The radius should be 1.43."
    assert clusterer.random_state is None, "The random state should be None by default."
    assert clusterer._estimator_type == "clusterer", "The estimator type should be 'clusterer'."

    labels = clusterer.fit_predict(X)

    assert labels is not None, "Labels should not be None after fit_predict."
    assert len(labels) == X.shape[0], "The number of labels should match the number of samples."
    assert len(set(labels)) <= X.shape[0], "The number of unique labels should not exceed the number of samples."

def test_radius_clustering_fit_predict_exact():
    """
    Test the fit_predict method of the RadiusClustering class with exact method.
    """
    clusterer = RadiusClustering(manner="exact", radius=1.43)

    assert clusterer.manner == "exact", "The manner should be 'exact'."
    assert clusterer.radius == 1.43, "The radius should be 1.43."
    assert clusterer.random_state is None, "The random state should be None by default."
    assert clusterer._estimator_type == "clusterer", "The estimator type should be 'clusterer'."

    labels = clusterer.fit_predict(X)

    assert labels is not None, "Labels should not be None after fit_predict."
    assert len(labels) == X.shape[0], "The number of labels should match the number of samples."
    assert len(set(labels)) <= X.shape[0], "The number of unique labels should not exceed the number of samples."

def test_radius_clustering_random_state():
    """
    Test the random state functionality of the RadiusClustering class.
    """
    clusterer = RadiusClustering(manner="approx", radius=1.43, random_state=123)

    assert clusterer.random_state == 123, "The random state should be set to 123."

    # Fit the model
    clusterer.fit(X)

    # Check that the random state is preserved
    assert clusterer.random_state == 123, "The random state should remain 123 after fitting."

    # Check that the results are consistent with the random state
    labels1 = clusterer.labels_

    # Re-initialize and fit again with the same random state
    clusterer2 = RadiusClustering(manner="approx", radius=1.43, random_state=123)
    clusterer2.fit(X)
    
    labels2 = clusterer2.labels_

    assert (labels1 == labels2).all(), "Labels should be consistent across runs with the same random state."

def test_deterministic_behavior():
    """
    Test the deterministic behavior of the RadiusClustering class with a fixed random state.
    """
    clusterer1 = RadiusClustering(manner="approx", radius=1.43, random_state=42)
    clusterer2 = RadiusClustering(manner="approx", radius=1.43, random_state=42)

    labels1 = clusterer1.fit_predict(X)
    labels2 = clusterer2.fit_predict(X)

    assert (labels1 == labels2).all(), "Labels should be the same for two instances with the same random state."

    clusterer1 = RadiusClustering(manner="exact", radius=1.43)
    clusterer2 = RadiusClustering(manner="exact", radius=1.43)
    labels1 = clusterer1.fit_predict(X)
    labels2 = clusterer2.fit_predict(X)
    assert (labels1 == labels2).all(), "Labels should be the same for two exact instances."
