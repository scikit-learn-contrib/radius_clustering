Usage
=====

Here's a basic example of how to use Radius Clustering:

.. code-block:: python

   from radius_clustering import RadiusClustering
   import numpy as np

   # Generate random data
   X = np.random.rand(100, 2)

   # Create an instance of MdsClustering
   rad = RadiusClustering(manner="approx", threshold=0.5)

   # Fit the model to the data
   rad.fit(X)

   # Get cluster labels
   labels = rad.labels_

   print(labels)