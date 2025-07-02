from sklearn.utils.estimator_checks import parametrize_with_checks
def test_import():
    import radius_clustering as rad


def test_from_import():
    from radius_clustering import RadiusClustering


from radius_clustering import RadiusClustering

@parametrize_with_checks([RadiusClustering()])
def test_check_estimator_api_consistency(estimator, check, request):

    """Check the API consistency of the RadiusClustering estimator
    """
    check(estimator)
