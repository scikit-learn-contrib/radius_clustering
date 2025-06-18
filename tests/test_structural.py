from logging import getLogger

logger = getLogger(__name__)
logger.setLevel("INFO")

def test_import():
    import radius_clustering as rad


def test_from_import():
    from radius_clustering import RadiusClustering

def test_check_estimator_api_consistency():
    from radius_clustering import RadiusClustering
    from sklearn.utils.estimator_checks import check_estimator

    # Check the API consistency of the RadiusClustering estimator
    check_estimator(RadiusClustering())
