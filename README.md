# Efficient Active Search

Efficient active search (EAS) is a simple method that extends learning-based construction heuristics with a powerful, iterative search phase. It is based on active search ([Bello et al., 2016](https://arxiv.org/abs/1611.09940)), which adjusts the weights of a (trained) model with respect to a single instance at test time using reinforcement learning. In contrast to the original active search, efficient active search only updates a subset of all (model) parameters during the search, while keeping all other parameters fixed. This drastically reduces the runtime without impairing the solution quality. We evaluate efficient active search on the traveling salesperson problem (TSP), the capacitated vehicle routing problem (CVRP) and the job shop scheduling problem (JSSP).

This repository contains an implementation of EAS for the POMO method of Kwon at al. which is a learning-based construction heuristic for TSP and CVRP instances. Furthermore, the repository contains the instances and models used in the experiments of our paper "Efficient Active Search for Combinatorial Optimization Problems" (https://openreview.net/forum?id=nO5caZwFwYu).

### Paper
```
@article{eas,
  title={Efficient active search for combinatorial optimization problems},
  author={Hottung, Andr{\'e} and Kwon, Yeong-Dae and Tierney, Kevin},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```


## Requirements

EAS requires python (>= 3.7) and the following python packages:

- numpy

- pytorch (we tested the code with version 1.8.0)

- tqdm


## Instances and Models

The directories *instances* and *trained_models* contain all used instances and all trained models, respectively.  The .pkl instances files have been generated using the generator from Kool et al. (available [here](https://github.com/wouterkool/dpdp)). The more realistic XE instances (in the CVRPLIB format) for the CVRP are from [here](https://github.com/ahottung/NLNS/tree/master/instances). The models for the TSP100 and CVRP100 have been trained by Kwon et al. and are available [here](https://github.com/yd-kwon/POMO/tree/master/OLD_ipynb_ver). The models for the XE instances have been trained by us. 


## Quick Start

The code implements 3 different EAS methods:

- EAS using embedding updates (`-method eas-emb`)
- EAS using added layer updates (`-method eas-lay`)
- EAS using tabular updates (`-method eas-tab`)

Furthermore, the the original active search (`-method as`) and sampling (`-method sampling`) are implemented as baselines. In all experiments we limited the search to 200 iterations (`-max_iter 200`).

Note that we performed our experiments on NVIDIA V100 GPUs with 32 GB of memory. You will need to reduce the batch size (set via `-batch_size`), if your GPU has less memory.

### TSP

To solve the instance set *tsp100_test_seed1234.pkl* using the model trained by [Kwon at al.](https://github.com/yd-kwon/POMO/tree/master/OLD_ipynb_ver) run the following commands:


```bash
# EAS-Emb
python3 run_search.py -problem TSP -method eas-emb -model_path  trained_models/TSP_100/ACTOR_state_dic.pt -instances_path instances/tsp/tsp100_test_seed1234.pkl -max_iter 200 -batch_size 150 -param_lr 0.0032 -param_lambda 0.0058

# EAS-Lay
python3 run_search.py -problem TSP -method eas-lay -model_path  trained_models/TSP_100/ACTOR_state_dic.pt -instances_path instances/tsp/tsp100_test_seed1234.pkl -max_iter 200 -batch_size 75 -param_lr 0.0032 -param_lambda 0.012

# EAS-Tab
python3 run_search.py -problem TSP -method eas-tab -model_path  trained_models/TSP_100/ACTOR_state_dic.pt -instances_path instances/tsp/tsp100_test_seed1234.pkl -max_iter 200 -batch_size 1000 -param_alpha 0.505 -param_sigma 8.57

# Active Search
python3 run_search.py -problem TSP -method as -model_path  trained_models/TSP_100/ACTOR_state_dic.pt -instances_path instances/tsp/tsp100_test_seed1234.pkl -max_iter 200 -batch_size 1 -param_lr 0.00026

# Sampling
python3 run_search.py -problem TSP -method sampling -model_path  trained_models/TSP_100/ACTOR_state_dic.pt -instances_path instances/tsp/tsp100_test_seed1234.pkl -max_iter 200 -batch_size 1000
```

###  CVRP

To solve the instance set *vrp100_test_seed1234.pkl* using the model trained by [Kwon at al.](https://github.com/yd-kwon/POMO/tree/master/OLD_ipynb_ver) run the following commands:

```bash
# EAS-Emb
python3 run_search.py -problem CVRP -method eas-emb -model_path  trained_models/CVRP_100/ACTOR_state_dic.pt -instances_path instances/cvrp/vrp100_test_seed1234.pkl -max_iter 200 -batch_size 150 -param_lr 0.0049 -param_lambda 0.0063

# EAS-Lay
python3 run_search.py -problem CVRP -method eas-lay -model_path  trained_models/CVRP_100/ACTOR_state_dic.pt -instances_path instances/cvrp/vrp100_test_seed1234.pkl -max_iter 200 -batch_size 50 -param_lr 0.0041 -param_lambda 0.013

# EAS-Tab
python3 run_search.py -problem CVRP -method eas-tab -model_path  trained_models/CVRP_100/ACTOR_state_dic.pt -instances_path instances/cvrp/vrp100_test_seed1234.pkl -max_iter 200 -batch_size 1000 -param_alpha 0.539 -param_sigma 9.55

# Active Search
python3 run_search.py -problem CVRP -method as -model_path  trained_models/CVRP_100/ACTOR_state_dic.pt -instances_path instances/cvrp/vrp100_test_seed1234.pkl -max_iter 200 -batch_size 1 -param_lr 2.6e-05

# Sampling
python3 run_search.py -problem CVRP -method sampling -model_path  trained_models/CVRP_100/ACTOR_state_dic.pt -instances_path instances/cvrp/vrp100_test_seed1234.pkl -max_iter 200 -batch_size 1000
```

### CVRP - XE Instances

The more realistic XE instances sets are solved with single instance search (i.e., the instances are solve sequentially, one by one) using the provided models trained by us.

For example, to solve the instances in *instances/cvrp/XE/XE_1* run:

```bash
# EAS-Emb
python3 run_search.py -problem CVRP -method eas-emb -model_path  trained_models/XE_1/ACTOR_state_dic.pt -instances_path instances/cvrp/XE/XE_1 -max_iter 200 -batch_size 1 -param_lr 0.0049 -param_lambda 0.0063 -round_distances -p_runs 10

# EAS-Lay
python3 run_search.py -problem CVRP -method eas-lay -model_path  trained_models/XE_1/ACTOR_state_dic.pt -instances_path instances/cvrp/XE/XE_1 -max_iter 200 -batch_size 1 -param_lr 0.0041 -param_lambda 0.013 -round_distances -p_runs 10

# EAS-Tab
python3 run_search.py -problem CVRP -method eas-tab -model_path  trained_models/XE_1/ACTOR_state_dic.pt -instances_path instances/cvrp/XE/XE_1 -max_iter 200 -batch_size 1 -param_alpha 0.539 -param_sigma 9.55 -round_distances -p_runs 10

# Active Search
python3 run_search.py -problem CVRP -method as -model_path  trained_models/XE_1/ACTOR_state_dic.pt -instances_path instances/cvrp/XE/XE_1 -max_iter 200 -batch_size 1 -param_lr 2.6e-05 -round_distances

# Sampling
python3 run_search.py -problem CVRP -method sampling -model_path  trained_models/XE_1/ACTOR_state_dic.pt -instances_path instances/cvrp/XE/XE_1 -max_iter 200 -batch_size 1 -round_distances -p_runs 10
```

If you do not have enough GPU memory, reduce the number of parallel runs per instance  (set via `-p_runs`) to a smaller value. To solve the other XE instance sets, you only have to change the model path and the instance path accordingly. Note that we use the same hyperparameters as for the CVRP100 instances.


## Acknowledgements

We implemented EAS for the [POMO method](https://arxiv.org/abs/2010.16011) using the [POMO code](https://github.com/yd-kwon/POMO/tree/master/OLD_ipynb_ver) made available by the authors. Note that we use the OLD_ipynb_ver version of the code. Furthermore, we use the models for the TSP100 and the CVRP100 trained by the POMO authors.

