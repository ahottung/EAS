# EAS

This repository **will soon** contain the code used for the experiments in the paper "Efficient Active Search for Combinatorial Optimization Problems" (https://openreview.net/forum?id=nO5caZwFwYu).

Efficient active search is a simple method that extends learning-based construction heuristics with a powerful, iterative search phase. It is based on active search (Bello et al., 2016), which adjusts the weights of a (trained) model with respect to a single instance at test time using reinforcement learning. In contrast to the original active search, efficient active search only updates a subset of all (model) parameters during the search, while keeping all other parameters fixed. This drastically reduces the runtime of  without impairing the solution quality. We evaluate efficient active search on the traveling salesperson problem (TSP), the capacitated vehicle routing problem (CVRP) and the job shop scheduling problem (JSSP).

### Paper
```
@article{eas,
  title={Efficient active search for combinatorial optimization problems},
  author={Hottung, Andr{\'e} and Kwon, Yeong-Dae and Tierney, Kevin},
  booktitle={International Conference on Machine Learning},
  year={2022}
}
```
