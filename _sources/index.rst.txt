.. _index:

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   details
   usage
   api
   auto_examples/index

Welcome to Radius Clustering's documentation!
=============================================


The Radius Clustering algorithm is a clustering under radius constraint algorithm. It is based on the minimum dominating set problem (MDS) in graph theory.

The algorithm is designed such that it can be used to cluster data points based on a radius constraint. The goal is to group data points such that the minimal maximum distance between any two points in the same cluster is less than or equal to a given radius.

The algorithm is based on the equivalence between the minimum dominating set problem and the clustering under radius constraint problem. The latter problem is characterized by a radius parameter :math:`r` and a set of points :math:`X`. The goal is to find a partition of the points into subsets such that each subset is contained in a ball of radius :math:`r`. Plus, the goal is to minimize the number of subsets.

This problem is proven to be NP-Hard, and the MDS problem is known to be NP-Hard as well.

We propose an implementation to tackle this specific problem, based upon the MDS problem. The idea is to use the MDS algorithm to find the representative points of each cluster, and then to assign each point to the nearest representative point.


.. warning:: Considering the NP-Hardness (or NP-Completeness) of the MDS problem, we alert that the overall complexity of any algorithm tackling this problem cannot be polynomial, unless P=NP. That is why we alert the user that the algorithm may take a long time to run on large datasets, especially when using the exact algorithm.
  From the experiments conducted, the exact algorithm is not recommended for datasets with more than 1000 points, but the overall complexity of the datasets and or the internal structure of the data may affect this threshold, in either way. For a more complete insight, we recommand the user to refer to the paper `Clustering under radius constraint using minimum dominating sets <https://hal.science/hal-04533921/>`_ or reading the :ref:`details` page of the documentation.



Acknowledgments
===============

The authors would like to thank the following people for their work that contributed either directly or indirectly to the development of this algorithm:

Authors & Contributors
----------------------

**Quentin Haenn**, ISAE-ENSMA, LIAS, France. PhD Student, first author of this work.

.. note::
    - `GitHub <https://github.com/quentinhaenn>`_
    - `Lab page <https://www.lias-lab.fr/fr/members/quentinhaenn/>`_
    - Mail: :email:`quentin.haenn@ensma.fr`

**Brice Chardin**, ISAE-ENSMA, LIAS, France. Associate Professor, co-author of this work.

.. note::
    - `Lab page <https://www.lias-lab.fr/fr/members/bricechardin/>`_

**Mickaël Baron**, ISAE-ENSMA, LIAS, France. Research Engineer, co-author of this work.

.. note::
    - `Lab page <https://www.lias-lab.fr/fr/members/mickaelbaron/>`_

Principal References
--------------------

.. [casado] A. Casado, S. Bermudo, A.D. López-Sánchez, J. Sánchez-Oro,
    An iterated greedy algorithm for finding the minimum dominating set in graphs,
    Mathematics and Computers in Simulation,
    Volume 207,
    2023
    Code available at https://github.com/AlejandraCasado

    *We rewrote the code in C++ to adapt to the need of python interfacing.*

.. [jiang] Jiang, Hua and Zheng, Zhifei, "An Exact Algorithm for the Minimum Dominating Set Problem", Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence,
    pages 5604--5612 -in- proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23, 2023, doi: 10.24963/ijcai.2023/622.
    Code available at https://github.com/huajiang-ynu.

    *We adapted the code to the need of python interfacing.*


.. [andersen] Jennie Andersen, Brice Chardin, Mohamed Tribak. "Clustering to the Fewest Clusters Under Intra-Cluster Dissimilarity Constraints". Proceedings of the 33rd IEEE International Conference on Tools with Artificial Intelligence, Nov 2021, Athens, Greece. pp.209-216, https://dx.doi.org/10.1109/ICTAI52525.2021.00036

.. [bien] Bien, J., & Tibshirani, R. (2011). Hierarchical Clustering with Prototypes via Minimax Linkage. http://faculty.marshall.usc.edu/Jacob-Bien/papers/jasa2011minimax.pdf

