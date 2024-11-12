## How it works

### Clustering under radius constraint
Clustering tasks are globally concerned about grouping data points into clusters based on some similarity measure. Clustering under radius constraints is a specific clustering task where the goal is to group data points such that the minimal maximum distance between any two points in the same cluster is less than or equal to a given radius. Mathematically, given a set of data points $X = \{x_1, x_2, \ldots, x_n\}$ and a radius $r$, the goal is to find a partition $ \mathcal{P}$ of $X$ into clusters $C_1, C_2, \ldots, C_k$ such that :
```math
\forall C \in \mathcal{P}, \min_{x_i \in C}\max_{x_j \in C} d(x_i, x_j) \leq r
```
where $d(x_i, x_j)$ is the dissimilarity between $x_i$ and $x_j$.

### Minimum Dominating Set (MDS) problem

The Radius Clustering package implements a clustering algorithm based on the Minimum Dominating Set (MDS) problem. The MDS problem is a well-known NP-Hard problem in graph theory, and it has been proven to be linked to the clustering under radius constraint problem. The MDS problem is defined as follows:

Given an undirected weighted graph $G = (V,E)$ where $V$ is a set of vertices and $E$ is a set of edges, a dominating set $D$ is a subset of $V$ such that every vertex in $V$ is either in $D$ or adjacent to a vertex in $D$. The goal is to find a dominating set $D$ such that the number of vertices in $D$ is minimized. This problem is known to be NP-Hard.

However, solving this problem in the context of clustering task can be useful, but we need some adaptations.

### Radius Clustering algorithm

To adapt the MDS problem to the clustering under radius constraint problem, we need to define a graph based on the data points. The vertices of the graph are the data points, and the edges are defined based on the distance between the data points. The weight of the edges is the dissimilarity between the data points. Then, the algorithm operates as follows:

1. Construct a graph $G = (V,E)$ based on the data points $X$.
2. Prune the graph by removing the edges $e_{ij}$ such that $d(x_i,x_j) > r$.
3. Solve the MDS problem on the pruned graph.
4. Assign each vertex to the closest vertex in the dominating set. In case of a tie, assign the vertex to the vertex with the smallest index.
5. Return the cluster labels.