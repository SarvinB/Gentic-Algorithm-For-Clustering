# Genetic Algorithm for Clustering (Iris Dataset)

An educational Python implementation of **genetic-algorithm-based clustering** using the [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). Instead of updating cluster centroids as in K-Means, this project searches over different assignments of data points to clusters using selection, crossover, and mutation.

> **Project status:** This repository is an experimental/educational implementation, **not a validated clustering benchmark**. The current code has known issues in mutation, distance calculation, and final solution selection. See [Known limitations](#known-limitations) before interpreting the output.

## Overview

The goal is to assign **150 Iris flowers** to **3 clusters** using their four numerical measurements:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

The dataset also includes species labels (*Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*). The labels are not required to calculate the clustering fitness; the notebook retains them for comparison and visualization.

### Project files

```text
Gentic-Algorithm-For-Clustering/
├── Genetic_Algorithm.py   # GA class, operators, and fitness calculation
├── Iris.ipynb             # Data preparation, GA execution, and plots
├── Iris.csv               # Iris dataset
└── README.md              # Project documentation (this file)
```

## Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook or JupyterLab

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib scikit-learn notebook
```

## Getting started

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/SarvinB/Gentic-Algorithm-For-Clustering.git
   cd Gentic-Algorithm-For-Clustering
   ```

2. Install the dependencies listed above.

3. Start Jupyter Notebook from the repository directory:

   ```bash
   jupyter notebook Iris.ipynb
   ```

4. Run the notebook cells in order. The notebook loads `Iris.csv`, creates an initial population, evolves it for 100 generations, and plots the resulting cluster assignments.

**Note:** The notebook uses random initial chromosomes and evolutionary operations without a fixed random seed. Results can differ between runs. The known implementation issues below can also affect the output.

## How it works

### 1. Prepare the data

The notebook:

1. Loads `Iris.csv` using pandas.
2. Shuffles the rows and converts species names to numerical labels.
3. Extracts the four flower measurements into `x`, excluding the `Id` and `Species` columns.
4. Standardizes `x` using `StandardScaler` so that Euclidean distances can be calculated on comparable feature scales.

The genetic algorithm uses **`x` (the standardized measurements)** to calculate distances. Although the `GA` constructor also receives `iris` and stores it as `self.iris`, that attribute is not used elsewhere in the class.

### 2. Encode candidate solutions

Each chromosome represents one possible assignment of the 150 flowers to three clusters. For example:

```text
Flower:      0  1  2  3  4  5
Chromosome:  0  2  1  1  0  2
```

Here, flowers 0 and 4 belong to cluster 0; flowers 2 and 3 to cluster 1; and flowers 1 and 5 to cluster 2.

In the supplied notebook, chromosomes have `n + 1` entries, and the last entry is set to `3`. That extra entry is **not** a flower assignment used in the clustering calculations.

### 3. Evolve the population

The notebook's default settings are:

| Parameter | Value |
| --- | ---: |
| Data points (`n`) | 150 |
| Clusters | 3 |
| Population size (`c_n`) | 30 |
| Selected parents (`p_n`) | 16 |
| Crossover cut positions | 30 |
| Generations (`t`) | 100 |

For each generation, the algorithm:

1. **Selects parents** by sampling indices at random from the current population.
2. **Performs crossover** by swapping segments between pairs of parent chromosomes, creating 16 children.
3. **Applies mutation** by attempting to reassign one randomly selected data point in each candidate chromosome.
4. **Calculates fitness** for all 46 candidates (30 existing chromosomes + 16 children).
5. **Selects survivors** by retaining the 30 highest-scoring candidates for the next generation.

### 4. Calculate fitness

The fitness function is intended to use the **average silhouette coefficient**, which rewards clustering assignments whose points are close to their own cluster and far from other clusters:

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

where:

- `a(i)` is the mean distance from point `i` to other points in its own cluster.
- `b(i)` is the minimum mean distance from point `i` to points in another cluster.

The chromosome's fitness is the average of these coefficients over the dataset. Higher values generally indicate better-separated clusters.

**Important:** The implementation has bugs in its distance calculations and does not reliably compute the textbook silhouette score as written.

### 5. Visualize results

The notebook plots the Iris measurements in two feature pairs:

- Sepal length vs. sepal width
- Petal length vs. petal width

The plots use colors to show species labels and/or candidate cluster assignments at different stages of the notebook. These are **two-dimensional views** of an algorithm that uses all four standardized features for distance calculations.

## Main methods

The `GA` class in `Genetic_Algorithm.py` contains:

| Method | Purpose |
| --- | --- |
| `parrent_choosing()` | Randomly choose parents for reproduction. |
| `uniform_cross_over()` | Create two children by alternating chromosome segments at random cut points. |
| `cross_over()` | Combine the existing population with offspring. |
| `mutation(x)` | Attempt to change a data point's cluster assignment. |
| `group_dist(index, cluster, x)` | Compute the mean Euclidean distance from a data point to a cluster. |
| `simularity(clusters, x)` | Calculate within-cluster distances. |
| `dissimularity(clusters, x)` | Calculate distances to other clusters. |
| `objective_function_simularity(x, index)` | Compute the silhouette-based fitness of one candidate. |
| `next_generation_choosing(x, y)` | Keep the highest-fitness chromosomes. |
| `winner(x, y)` | Attempt to return the final solution and an accuracy-like measure. |

The spelling of some method names (`parrent`, `simularity`, `dissimularity`) follows the original source code.

## Known limitations

The current implementation should be treated as a starting point for study and improvement:

1. **Unused constructor data:** `self.iris` is assigned in `GA.__init__()` but not used later. The actual feature matrix is passed separately as `x` to distance, mutation, and fitness methods.
2. **Mutation bug:** `mutation()` initializes `mini = math.inf` but never updates `mini` when a closer cluster is found. Consequently, the comparison does not reliably select the closest cluster; with finite distances, the chosen gene ends up assigned to the last cluster checked.
3. **Distance indexing bug:** `group_dist()` checks `if i != index`, comparing a position *within the cluster list* with a *dataset row index*. The intended self-distance exclusion should instead compare `cluster[i]` with `index`.
4. **Silhouette edge cases:** Empty clusters, singleton clusters, and zero denominators require explicit treatment. The current fitness implementation does not reliably handle all such cases.
5. **Final winner mismatch:** `winner()` computes scores by indexing `self.population`, then retrieves the chosen chromosome from `self.chromosomes`. These are different collections after survivor selection, so the returned chromosome may not match the one scored as best.
6. **Unreliable accuracy:** Direct comparison of cluster IDs with species IDs (using a fixed shift) is not a valid general clustering-accuracy measure because cluster IDs are arbitrary. Prefer permutation-invariant metrics such as the **Adjusted Rand Index (ARI)** or **Normalized Mutual Information (NMI)**. Species labels should be used only for external evaluation, not for optimizing unsupervised clustering.
7. **Hard-coded settings:** Several parts of the implementation assume exactly three clusters and 150 data points.

## Possible improvements

- Fix the mutation and distance-calculation logic.
- Store the feature matrix in the GA class to avoid passing `x` into many methods.
- Remove the unused trailing chromosome element unless it has a defined purpose.
- Handle empty and singleton clusters robustly.
- Use a verified silhouette-score implementation to validate the fitness function.
- Fix final winner selection and report permutation-invariant evaluation metrics.
- Add a random seed, convergence history, unit tests, and comparisons with K-Means.

## References

- [Original project repository](https://github.com/SarvinB/Gentic-Algorithm-For-Clustering)
- [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [scikit-learn: Silhouette Coefficient](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [scikit-learn: Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)

---

*This README documents the code as it currently exists; it does not imply that the implementation issues have been fixed.*
