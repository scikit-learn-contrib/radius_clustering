Usage
=====

This page provides a quick guide on how to use the `radius_clustering` package for clustering tasks. The package provides a simple interface for performing radius-based clustering on datasets based on the Minimum Dominating Set (MDS) algorithm.

This page is divided into three main sections:
1. **Basic Usage**: A quick example of how to use the `RadiusClustering` class and perform clustering with several parameters.
2. **Custom Dissimilarity Function**: How to use a custom dissimilarity function with the `RadiusClustering` class.
3. **Custom MDS Solver**: How to implement a custom MDS solver for more advanced clustering tasks, eventually with less guarantees on the results.


Basic Usage
-----------------

The `RadiusClustering` class provides a straightforward way to perform clustering based on a specified radius. You can choose between an approximate or exact method for clustering, depending on your needs.

Here's a basic example of how to use Radius Clustering with the `RadiusClustering` class, using the approximate method:

.. code-block:: python

   from radius_clustering import RadiusClustering
   import numpy as np

   # Generate random data
   X = np.random.rand(100, 2)

   # Create an instance of MdsClustering
   rad = RadiusClustering(manner="approx", radius=0.5) 
   # Attention: the 'threshold' parameter is deprecated by version 1.3.0
   # and will be removed in a future version. Use 'radius' instead.

   # Fit the model to the data
   rad.fit(X)

   # Get cluster labels
   labels = rad.labels_

   print(labels)

Similarly, you can use the exact method by changing the `manner` parameter to `"exact"`:
.. code-block:: python
   # [...] Exact same code as above
   rad = RadiusClustering(manner="exact", radius=0.5) #change this parameter
   # [...] Exact same code as above

Custom Dissimilarity Function
-----------------------------

The main reason behind the `radius_clustering` package is that users eventually needs to use a dissimilarity function that is not a metric (or distance) function. Plus, sometimes context requires a domain-specific dissimilarity function that is not provided by default, and needs to be implemented by the user.

To use a custom dissimilarity function, you can pass it as a parameter to the `RadiusClustering` class. Here's an example of how to do this:
.. code-block:: python

   from radius_clustering import RadiusClustering
   import numpy as np

   # Generate random data
   X = np.random.rand(100, 2)

   # Define a custom dissimilarity function
   def dummy_dissimilarity(x, y):
       return np.linalg.norm(x - y) + 0.1  # Example: add a constant to the distance

   # Create an instance of MdsClustering with the custom dissimilarity function
   rad = RadiusClustering(manner="approx", radius=0.5, metric=dummy_dissimilarity)

   # Fit the model to the data
   rad.fit(X)

   # Get cluster labels
   labels = rad.labels_

   print(labels)


.. note::
   The custom dissimilarity function will be passed to scikit-learn's `pairwise_distances` function, so it should be compatible with the expected input format and return type. See the scikit-learn documentation for more details on how to implement custom metrics.

Custom MDS Solver
-----------------

The two default solvers provided by the actual implementation of the `radius_clustering` package are focused on exactness (or proximity to exactness) of the results of a NP-hard problem. So, they may not be suitable for all use cases, especially when performance is a concern.
If you have your own implementation of a Minimum Dominating Set (MDS) solver, you can use it with the `RadiusClustering` class ny using the :py:func:'RadiusClustering.set_solver' method. It will check that the solver is compatible with the expected input format and return type, and will use it to perform clustering.

Here's an example of how to implement a custom MDS solver and use it with the `RadiusClustering` class, using NetworkX implementation of the dominating set problem : 

.. code-block:: python

   from radius_clustering import RadiusClustering
   import time
   import numpy as np
   import networkx as nx

   # Generate random data
   X = np.random.rand(100, 2)

   # Define a custom MDS solver using NetworkX
   def custom_mds_solver(n, edges, nb_edges, random_state=None):
      start = time.time()
      graph = nx.Graph(edges)
      centers = list(nx.algorithms.dominating_set(graph))
      centers.sort()
      end = time.time()
      return centers, end - start

   # Create an instance of MdsClustering with the custom MDS solver
   rad = RadiusClustering(manner="approx", radius=0.5)
   rad.set_solver(custom_mds_solver)

   # Fit the model to the data
   rad.fit(X)

   # Get cluster labels
   labels = rad.labels_

   print(labels)

.. note::
   The custom MDS solver should accept the same parameters as the default solvers, including the number of points `n`, the edges of the graph `edges`, the number of edges `nb_edges`, and an optional `random_state` parameter for reproducibility. It should return a list of centers and the time taken to compute them.
   The `set_solver` method will check that the custom solver is compatible with the expected input format and return type, and will use it to perform clustering.
   If the custom solver is not compatible, it will raise a `ValueError` with a descriptive message.

.. attention::
   We cannot guarantee that the custom MDS solver will produce the same results as the default solvers, especially if it is not purposely designed to solve the Minimum Dominating Set problem but rather just finds a dominating set. The results may vary depending on the implementation and the specific characteristics of the dataset.
   As an example, a benchmark of our solutions and a custom one using NetworkX is available in the `Example Gallery` section of the documentation, which shows that the custom solver may produce different results than the default solvers, especially in terms of the number of clusters and the time taken to compute them.
   However, it can be useful for specific use cases where performance is a concern or when you have a custom implementation that fits your needs better.

