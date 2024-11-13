def test_imports():
    import radius_clustering as rad


def test_from_import():
    from radius_clustering import RadiusClustering


def test_radius_clustering_approx():
    from radius_clustering import RadiusClustering
    from sklearn import datasets

    # Load the Iris dataset
    iris = datasets.fetch_openml(name="iris", version=1, parser="auto")
    X = iris["data"]  # Use dictionary-style access instead of attribute access

    graph_mds_api_consistent = RadiusClustering(manner="approx", threshold=1.43)

    result_api_consistent = graph_mds_api_consistent.fit_predict(X)


def test_radius_clustering_exact():
    from radius_clustering import RadiusClustering
    from sklearn import datasets

    # Load the Iris dataset
    iris = datasets.fetch_openml(name="iris", version=1, parser="auto")
    X = iris["data"]  # Use dictionary-style access instead of attribute access

    graph_mds_api_consistent = RadiusClustering(manner="exact", threshold=1.43)

    result_api_consistent = graph_mds_api_consistent.fit_predict(X)
