import pytest
import numpy as np
from radius_clustering import RadiusClustering
from sklearn.datasets import load_iris

@pytest.fixture
def iris_data():
    """Fixture to load the Iris dataset."""
    data = load_iris()
    return data.data

@pytest.fixture
def approx_results():
    """Fixture to store results for approximate clustering."""
    results = {
        'labels': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                   0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                   1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,2,2,2,1,2,2,2,2,
                   2,2,1,1,2,2,2,2,1,2,1,2,1,2,2,1,1,2,2,2,2,2,1,2,2,2,2,1,2,2,2,1,2,2,2,1,2,
                   2,1],
        "centers": [0,96,125],
        "time" : 0.0280,
        "effective_radius": 1.4282856857085722
    }
    return results

@pytest.fixture
def exact_results():
    """Fixture to store results for exact clustering."""
    results = {
        'labels':[
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,2,2,2,2,1,2,2,2,2,
            2,2,1,1,2,2,2,2,1,2,1,2,1,2,2,1,1,2,2,2,2,2,1,2,2,2,2,1,2,2,2,1,2,2,2,1,2,
            2,1
                ],
        "centers": [0, 96, 102],
        "time": 0.0004,
        "effective_radius": 1.4282856857085722
    }
    return results

def assert_results_exact(results, expected):
    """Helper function to assert clustering results."""
    assert_results(results, expected)
    assert set(results.labels_) == set(expected['labels']), "Labels do not match expected"
    assert results.centers_ == expected['centers'], "Centers do not match expected"
    assert np.sum(results.labels_ - expected['labels']) == 0, "Labels do not match expected"

def assert_results(results, expected):
    assert len(results.labels_) == len(expected['labels']), "Labels length mismatch"
    assert abs(results.mds_exec_time_ - expected['time']) < 0.1, "Execution time mismatch by more than 10%"
    assert abs(results.effective_radius_ - expected['effective_radius'])/results.effective_radius_ < 0.1, "Effective radius mismatch"

def test_exact(iris_data, exact_results):
    """Test the RadiusClustering with exact"""
    clustering = RadiusClustering(radius=1.43, manner='exact').fit(iris_data)
    assert_results_exact(clustering, exact_results)

def test_approx(iris_data, approx_results):
    """Test the RadiusClustering with approx."""
    clustering = RadiusClustering(radius=1.43, manner='approx').fit(iris_data)
    assert_results(clustering, approx_results)
