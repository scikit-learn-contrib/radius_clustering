from radius_clustering import RadiusClustering
import pytest

def test_symmetric():
    """
    Test that the RadiusClustering class can handle symmetric distance matrices.
    """
    import numpy as np

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


def test_fit():
    """
    Test that the RadiusClustering class can fit to a distance matrix and to a feature matrix.
    This test checks both the exact and approximate methods of clustering.
    """
    import numpy as np

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

    # Create a feature matrix
    X_features = np.array([[0, 1],
                           [1, 0],
                           [2, 1]])

    clustering = RadiusClustering(manner="approx", radius=1.5)
    clustering.fit(X_features)

    # Check that the labels are assigned correctly
    assert len(clustering.labels_) == X_features.shape[0], "Labels length should match number of samples."
    assert clustering.nb_edges_ > 0, "There should be edges in the graph."
    assert clustering._check_symmetric(clustering.dist_mat_), "Distance matrix should be symmetric after computed from features."

def test_radius_clustering_invalid_manner():
    """
    Test that an error is raised when an invalid manner is provided.
    """
    with pytest.raises(ValueError, match="Invalid manner. Choose either 'exact' or 'approx'."):
        RadiusClustering(manner="invalid", radius=1.43).fit([[0, 1], [1, 0], [2, 1]])

    with pytest.raises(ValueError, match="Invalid manner. Choose either 'exact' or 'approx'."):
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