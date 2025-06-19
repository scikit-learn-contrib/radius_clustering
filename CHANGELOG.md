# Changelog

All notable changes to this project will be documented in this file.

## [1.4.0] - 2025-06-19

### Contributors

- [@quentinhaenn](Quentin Haenn) - Main developer and maintainer

### Added

- Added support for custom MDS solvers in the `RadiusClustering` class.
- Updated the documentation to include examples of using custom MDS solvers.
- Added more examples and tutorials to the documentation.

### Changed

- Improved documentation and examples for the `RadiusClustering` class.
- Updated the README to reflect the new features and improvements in version 1.4.0
- Updated the test cases to ensure compatibility with the new features.
- Refactored the main codebase to improve readability and maintainability.
- Prepared the codebase for future adds of MDS solvers and/or clustering algorithms.

## [1.3.0] - 2025-06-18

### Contributors

- [@quentinhaenn](Quentin Haenn) - Main developer and maintainer

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

## [1.2.0] - 2024-10

### Contributors

- [@quentinhaenn](Quentin Haenn) - Main developer and maintainer
- [@mickaelbaron](Mickaël Baron) - Contributor and maintainer

### Added

- Added CI/CD pipelines with GitHub Actions for automated testing and deployment.
- Added package metadata for better integration with PyPI.
- Added a badge for the GitHub Actions workflow status in the README.
- Added a badge for the Python version supported in the README.
- Added a badge for the code style (Ruff) in the README.
- Added a badge for the license in the README.
- Added CI/CD pipelines for PyPI deployment (including test coverage, compiling extensions and wheels, and uploading to PyPI).
- Resolving issues with compiling Cython extensions on Windows and MacOS.
