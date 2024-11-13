.. _installation:

============
Installation
============

There are different ways to install Radius Clustering:

* :ref:`From PyPI <installation-pypi>`. This is the recommended way to install Radius Clustering. It will provide a stable version and pre-built packages are available for most platforms.

* :ref:`From the source <installation-source>`. This is best for users who want the latest features and are comfortable building from source. This is also needed if you want to contribute to the project.

.. _installation-pypi:

Installing from PyPI
--------------------

.. raw:: html

  <style>
    /* Show caption on large screens */
    @media screen and (min-width: 960px) {
      .install-instructions .sd-tab-set {
        --tab-caption-width: 20%;
      }

      .install-instructions .sd-tab-set.tabs-os::before {
        content: "Operating System";
      }

      .install-instructions .sd-tab-set.tabs-package-manager::before {
        content: "Package Manager";
      }
    }
  </style>

.. div:: install-instructions

  .. tab-set::
    :class: tabs-os

    .. tab-item:: Windows
      :class-label: tab-4

          Install the 64-bit version of Python 3, for instance from the
          `official website <https://www.python.org/downloads/windows/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install Radius Clustering.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: powershell

            python -m venv rad-env
            rad-env\Scripts\activate  # activate
            pip install -U radius-clustering

          In order to check your installation, you can use:

          .. prompt:: powershell

            python -m pip show radius-clustering # show radius-clustering version and location
            python -m pip freeze             # show all installed packages in the environment

    .. tab-item:: MacOS
      :class-label: tab-4


          Install Python 3 using `homebrew <https://brew.sh/>`_ (`brew install python`)
          or by manually installing the package from the `official website
          <https://www.python.org/downloads/macos/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install Radius Clustering.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash

            python -m venv rad-env
            source rad-env/bin/activate  # activate
            pip install -U radius-clustering

          In order to check your installation, you can use:

          .. prompt:: bash

            python -m pip show radius-clustering  # show radius-clustering version and location
            python -m pip freeze             # show all installed packages in the environment

    .. tab-item:: Linux
      :class-label: tab-4

          Python 3 is usually installed by default on most Linux distributions. To
          check if you have it installed, try:

          .. prompt:: bash

            python3 --version
            pip3 --version

          If you don't have Python 3 installed, please install `python3` and
          `python3-pip` from your distribution's package manager.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install Radius Clustering.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash

            python3 -m venv rad-env
            source rad-env/bin/activate  # activate
            pip3 install -U radius-clustering

          In order to check your installation, you can use:

          .. prompt:: bash

            python3 -m pip show radius-clustering  # show radius-clustering version and location
            python3 -m pip freeze             # show all installed packages in the environment


Using an isolated environment such as pip venv or conda makes it possible to
install a specific version of mds-clustering with pip or conda and its dependencies
independently of any previously installed Python packages. In particular under Linux
it is discouraged to install pip packages alongside the packages managed by the
package manager of the distribution (apt, dnf, pacman...).

Note that you should always remember to activate the environment of your choice
prior to running any Python command whenever you start a new terminal session.

If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).


.. _installation-source:

Installing from the source
--------------------------

To install Radius Clustering from the source, you need to clone the repository and
install the package using the following commands:

.. prompt:: bash

  cd /path/to/your/folder
  python -m venv rad-env
  source rad-env/bin/activate  # activate
  python -m pip install .

To check your installation, you can use:

.. prompt:: bash

  python -m pip show radius-clustering  # show radius-clustering version and location
  python -m pip freeze             # show all installed packages in the environment
  python -c "from radius_clustering import *; rad = RadiusClustering(); print(rad)"

If you want to contribute to the project, you will need to install the development
dependencies. You can do this by running:

.. prompt:: bash

  python -m pip install -e .[dev]

Alternatively, if you want to contribute only to the documentation, you can install
the documentation dependencies by running:

.. prompt:: bash

  python -m pip install -e .[docs]

Dependencies
++++++++++++


The minimum version of radius-clustering dependencies are listed below along with its
purpose.

.. list-table::
    :header-rows: 1

    * - Dependency
      - Minimum version
      - Purpose
    * - numpy
      - 1.23.4
      - Build, Install
    * - scipy
      - 1.12.0
      - Build, Install
    * - scikit-learn
      - 1.2.2
      - Build, Install
    * - cython
      - 3.0.10
      - Build
    * - setuptools
      - 61.0.0
      - Build
    * - pytest
      - 8.3.3
      - Tests
    * - ruff
      - 0.2.1
      - Tests
    * - black
      - 24.3.0
      - Tests
    * - matplotlib
      - 3.6.2
      - Docs, Examples
    * - sphinx
      - 8.1.3
      - Docs
    * - sphinx-copybutton
      - 0.5.2
      - Docs
    * - sphinx-rtd-theme
      - 3.0.0
      - Docs
    * - sphinx_design
      - 0.6.1
      - Docs
    * - sphinx_gallery
      - 0.18.0
      - Docs
    * - sphinx-prompt
      - 1.9.0
      - Docs
