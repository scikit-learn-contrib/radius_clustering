### Experimental results

The Radius Clustering package provides two algorithms to solve the MDS problem: an exact algorithm and an approximate algorithm. The approximate algorithm is based on a heuristic that iteratively selects the vertex that dominates the most vertices in the graph. The exact algorithm is based on a branch-and-bound algorithm that finds the minimum dominating set in the graph. Experimentation has been conducted on real-world datasets to compare the performances of these two algorithms, and compare them to state-of-the-art clustering algorithms. The complete results are available in the paper [Clustering under radius constraint using minimum dominating sets](https://hal.science/hal-04533921/).

The algorithms selected for comparison are:

1. Equiwide clustering (EQW-LP), a state-of-the-art exact algorithm using LP formulation of the problem [[3]](https://hal.science/hal-03356000)
2. ProtoClust [[4](http://faculty.marshall.usc.edu/Jacob-Bien/papers/jasa2011minimax.pdf)]

Here are some key results from the experiments:

Table 1: Average running time (in seconds) of the algorithms on real-world datasets.

| **Dataset**              | **MDS-APPROX** | **MDS-EXACT** | **EQW-LP**   | **PROTOCLUST** |
|--------------------------|----------------|---------------|--------------|----------------|
| **Iris**                 | 0.062 ± 0.01   | 0.009 ± 0.00  | 0.018 ± 0.01 | 0.026 ± 0.00   |
| **Wine**                 | 0.029 ± 0.00   | 0.010 ± 0.00  | 0.014 ± 0.00 | 0.034 ± 0.00   |
| **Glass Identification** | 0.015 ± 0.00   | 0.020 ± 0.00  | 0.026 ± 0.00 | 0.046 ± 0.00   |
| **Ionosphere**           | 0.078 ± 0.01   | 2.640 ± 0.05  | 0.104 ± 0.00 | 0.120 ± 0.00   |
| **WDBC**                 | 0.315 ± 0.01   | 0.138 ± 0.00  | 0.197 ± 0.01 | 0.402 ± 0.00   |
| **Synthetic Control**    | 0.350 ± 0.03   | 0.036 ± 0.00  | 0.143 ± 0.01 | 0.489 ± 0.00   |
| **Vehicle**              | 0.955 ± 0.04   | 0.185 ± 0.00  | 0.526 ± 0.01 | 0.830 ± 0.01   |
| **Yeast**                | 2.361 ± 0.03   | 738.8 ± 0.30  | 6.718 ± 0.02 | 2.374 ± 0.08   |
| **Ozone**                | 49.82 ± 1.18   | 1447 ± 0.54   | 26.86 ± 0.63 | 15.32 ± 0.15   |
| **Waveform**             | 48.01 ± 0.39   | 8813 ± 57.80  | 233.9 ± 1.45 | 61.27 ± 0.08   |

Table 2: Number of clusters obtained on real-world datasets.

| **Dataset**              | **MDS-APPROX** | **MDS-EXACT** | **EQW-LP** | **PROTOCLUST** |
|--------------------------|----------------|---------------|------------|----------------|
| **Iris**                 | 3              | 3             | 3          | 4              |
| **Wine**                 | 4              | 3             | 3          | 4              |
| **Glass Identification** | 7              | 6             | 6          | 7              |
| **Ionosphere**           | 2              | 2             | 2          | 5              |
| **WDBC**                 | 2              | 2             | 2          | 3              |
| **Synthetic Control**    | 8              | 6             | 6          | 8              |
| **Vehicle**              | 5              | 4             | 4          | 6              |
| **Yeast**                | 10             | 10            | 10         | 13             |
| **Ozone**                | 3              | 2             | 2          | 3              |
| **Waveform**             | 3              | 3             | 3          | 6              |


Table 3: Compactness of the clusters (maximal radius obtained after clustering) obtained on real-world datasets.

| **Dataset**              | **MDS-APPROX** | **MDS-EXACT** | **EQW-LP** | **PROTOCLUST** |
|--------------------------|----------------|---------------|------------|----------------|
| **Iris**                 | 1.43           | 1.43          | 1.43       | 1.24           |
| **Wine**                 | 220.05         | 232.08        | 232.08     | 181.35         |
| **Glass Identification** | 3.94           | 3.94          | 3.94       | 3.31           |
| **Ionosphere**           | 4.45           | 5.45          | 5.45       | 5.35           |
| **WDBC**                 | 1197.42        | 1197.42       | 1197.42    | 907.10         |
| **Synthetic Control**    | 66.59          | 70.11         | 70.11      | 68.27          |
| **Vehicle**              | 150.87         | 155.05        | 155.05     | 120.97         |
| **Yeast**                | 0.42           | 0.42          | 0.42       | 0.42           |
| **Ozone**                | 235.77         | 245.58        | 245.58     | 194.89         |
| **Waveform**             | 10.73          | 10.73         | 10.73      | 10.47          |


#### Key insights:

- The approximate algorithm is significantly faster than the exact algorithm, but it may not always provide the optimal solution.
- The exact algorithm is slower but provides the optimal solution. Does not scale well to large datasets, due to the NP-Hard nature of the problem.
- The approximate algorithm is a good trade-off between speed and accuracy for most datasets.
- MDS based approach are both more accurate than Protoclust. However, Protoclust is remarkably faster on most datasets.


> :memo: **Note**: The results show that MDS-based clustering algorithms might be a good alternative to state-of-the-art clustering algorithms for clustering under radius constraint problems.

> :memo: **Note**: Since the publication of the paper, the Radius Clustering package has been improved and optimized. The results presented here are based on the initial version of the package. For the latest results, please refer to the documentation or the source code.


## References

- [3] [Clustering to the fewest clusters under intra-cluster dissimilarity constraints](https://hal.science/hal-03356000)
- [4] [Hierarchical Clustering with prototypes via Minimax Linkage](http://faculty.marshall.usc.edu/Jacob-Bien/papers/jasa2011minimax.pdf)
 
