# Contributing to Radius Clustering

First off, thank you for considering contributing to Radius Clustering! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/scikit-learn-contrib/radius_clustering/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If you've decided to contribute, you'll need to fork the repository and create a new branch.

```bash
git checkout -b my-new-feature
```

## Getting started

To get started with the development, you need to install the package in an editable mode with all the development dependencies. It is highly recommended to do this in a virtual environment.

```bash
pip install -e ".[dev]"
```

This will install the package and all the tools needed for testing and linting.

## Running Tests

To ensure that your changes don't break anything, please run the test suite.

```bash
pytest
```

## Code Style

This project uses `ruff` for linting and `black` for formatting. We use `pre-commit` to automatically run these tools before each commit.

To set up `pre-commit`, run:

```bash
pre-commit install
```

This will ensure your contributions match the project's code style.

## Submitting a Pull Request

When you're ready to submit your changes, please write a clear and concise pull request message. Make sure to link any relevant issues.

Thank you for your contribution!
