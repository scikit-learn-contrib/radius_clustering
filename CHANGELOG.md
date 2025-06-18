# Changelog

## [1.2.3] - 2025-06-18

### Added

- Full test coverage for the entire codebase.
- Badge for test coverage in the README.
- Added `radius` parameter to the `RadiusClustering` class, allowing users to specify the radius for clustering.

### Deprecated

- Deprecated the `threshold` parameter in the `RadiusClustering` class. Use `radius` instead.

### Changed

- Updated all the attributes in the `RadiusClustering` class to fit `scikit-learn` standards and conventions.
- Updated the tests cases to reflect the changes in the `RadiusClustering` class.
- Updated README and documentation to reflect the new `radius` parameter and the deprecation of `threshold`.
