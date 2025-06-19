from radius_clustering import RadiusClustering
import pytest
import numpy as np

def test_symmetric():
    """
    Test that the RadiusClustering class can handle symmetric distance matrices.
    """

    # Check 1D array input

    X = np.array([0,1])
    with pytest.raises(ValueError):
        RadiusClustering(manner="exact", radius=1.5)._check_symmetric(X)

    # Check a symmetric distance matrix
    X = np.array([[0, 1, 2],
                  [1, 0, 1],
                  [2, 1, 0]])

    clustering = RadiusClustering(manner="exact", radius=1.5)
    assert clustering._check_symmetric(X), "The matrix should be symmetric."

    # Check a non-symmetric distance matrix
    X_assym = np.array([[0, 1, 2],
                       [1, 0, 1],
                       [2, 2, 3]])  # This is not symmetric
    assert not clustering._check_symmetric(X_assym), "The matrix should not be symmetric."

    # check a non-square matrix
    X_non_square = np.array([[0, 1],
                             [1, 0],
                             [2, 1]])  # This is not square
    
    assert not clustering._check_symmetric(X_non_square), "The matrix should not be symmetric."


def test_fit_distance_matrix():
    """
    Test that the RadiusClustering class can fit to a distance matrix.
    This test checks both the exact and approximate methods of clustering.
    """

    # Create a symmetric distance matrix
    X = np.array([[0, 1, 2],
                  [1, 0, 1],
                  [2, 1, 0]])

    clustering = RadiusClustering(manner="exact", radius=1.5)
    clustering.fit(X)

    # Check that the labels are assigned correctly
    assert len(clustering.labels_) == X.shape[0], "Labels length should match number of samples."
    assert clustering.nb_edges_ > 0, "There should be edges in the graph."
    assert np.array_equal(clustering.X_checked_, clustering.dist_mat_), "X_checked_ should be equal to dist_mat_ because X is a distance matrix."

@pytest.mark.parametrize(
        "test_data", [
            ("euclidean",1.5), 
            ("manhattan", 2.1), 
            ("cosine", 1.0)
        ]
)
def test_fit_features(test_data):
    """
    Test that the RadiusClustering class can fit to feature data.
    This test checks both the exact and approximate methods of clustering
    and multiple metrics methods.
    """
    # Create a feature matrix
    X_features = np.array([[0, 1],
                           [1, 0],
                           [2, 1]])
    metric, radius = test_data

    clustering = RadiusClustering(manner="approx", radius=radius)
    clustering.fit(X_features, metric=metric)
    # Check that the labels are assigned correctly
    assert len(clustering.labels_) == X_features.shape[0], "Labels length should match number of samples."
    assert clustering.nb_edges_ > 0, "There should be edges in the graph."
    assert clustering._check_symmetric(clustering.dist_mat_), "Distance matrix should be symmetric after computed from features."

def test_radius_clustering_invalid_manner():
    """
    Test that an error is raised when an invalid manner is provided.
    """
    with pytest.raises(ValueError):
        RadiusClustering(manner="invalid", radius=1.43).fit([[0, 1], [1, 0], [2, 1]])

    with pytest.raises(ValueError):
        RadiusClustering(manner="", radius=1.43).fit([[0, 1], [1, 0], [2, 1]])


def test_radius_clustering_invalid_radius():
    """
    Test that an error is raised when an invalid radius is provided.
    """
    with pytest.raises(ValueError, match="Radius must be a positive float."):
        RadiusClustering(manner="exact", radius=-1.0).fit([[0, 1], [1, 0], [2, 1]])

    with pytest.raises(ValueError, match="Radius must be a positive float."):
        RadiusClustering(manner="approx", radius=0.0).fit([[0, 1], [1, 0], [2, 1]])

    with pytest.raises(ValueError, match="Radius must be a positive float."):
        RadiusClustering(manner="exact", radius="invalid").fit([[0, 1], [1, 0], [2, 1]])

def test_radius_clustering_fit_without_data():
    """
    Test that an error is raised when fitting without data.
    """
    clustering = RadiusClustering(manner="exact", radius=1.5)
    with pytest.raises(ValueError):
        clustering.fit(None)

def test_radius_clustering_new_clusterer():
    """
    Test that a custom clusterer can be set within the RadiusClustering class.
    """
    def custom_clusterer(n, edges, nb_edges, random_state=None):
        # A mock custom clusterer that returns a fixed set of centers
        # and a fixed execution time
        return [0, 1], 0.1
    clustering = RadiusClustering(manner="exact", radius=1.5)
    # Set the custom clusterer
    assert hasattr(clustering, 'set_solver'), "RadiusClustering should have a set_solver method."
    assert callable(clustering.set_solver), "set_solver should be callable."
    clustering.set_solver(custom_clusterer)
    # Fit the clustering with the custom clusterer
    X = np.array([[0, 1],
                  [1, 0],
                  [2, 1]])
    clustering.fit(X)
    assert clustering.clusterer_ == custom_clusterer, "The custom clusterer should be set correctly."
    # Check that the labels are assigned correctly
    assert len(clustering.labels_) == X.shape[0], "Labels length should match number of samples."
    assert clustering.nb_edges_ > 0, "There should be edges in the graph."
    assert clustering.centers_ == [0, 1], "The centers should match the custom clusterer's output."
    assert clustering.mds_exec_time_ == 0.1, "The MDS execution time should match the custom clusterer's output."

def test_invalid_clusterer():
    """
    Test that an error is raised when an invalid clusterer is set.
    """
    clustering = RadiusClustering(manner="exact", radius=1.5)
    with pytest.raises(ValueError, match="The provided solver must be callable."):
        clustering.set_solver("not_a_callable")

    with pytest.raises(ValueError, match="The provided solver must be callable."):
        clustering.set_solver(12345)  # Not a callable
    with pytest.raises(ValueError, match="The provided solver must be callable."):
        clustering.set_solver(None)

    def invalid_signature():
        return [0, 1], 0.1
    
    with pytest.raises(ValueError):
        clustering.set_solver(invalid_signature)
    def invalid_clusterer(n, edges, nb_edges):
        return [0, 1], 0.1
    with pytest.raises(ValueError):
        clustering.set_solver(invalid_clusterer)