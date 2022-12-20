# COMP 138 RL Final Project: Implementing Cross-Embodiment Inverse Reinforcement Learning
Anirudh Nair, Binit Shrestha

- [Overview](#overview)
- [Setup](#setup)
- [Datasets](#datasets)
- [Code Navigation](#code-navigation)
- [Experiments: Reproducing Paper Results](#experiments-reproducing-paper-results)
- [Acknowledgments](#acknowledgments)


## Overview

Robot learning from third person videos to complete tasks has been a long standing goal of the research community. However, third person videos introduce another problem: the cross-embodiment problem. Human dynamics are different to robot dynamics, and different agents may have different action spaces to accomplish the same task. In this work, we investigate cross-embodiment inverse reinforcement learning (X-IRL), a method to overcome the cross-embodiment problem by employing temporal cycle-consistency to learn an agent-agnostic reward, from which a reinforcement learning algorithm can use to obtain a policy. We implement X-IRL on the X-MAGICAL Benchmark, an environment suite to test and develop cross-embodiment methods. Additionally, we develop the beginnings of a graph-abstraction extension to X-IRL that can represent the entire scene as a graph, lowering the input dimensions to the temporal cycle-consistency.

## Setup

We use Python 3.8 and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for development. To create an environment and install dependencies, run the following steps inside the root directory:

```bash
# Create and activate environment.
conda create -n xirl python=3.8
conda activate xirl

# Install dependencies.
pip install -r requirements.txt
```

## Datasets + Experiments Download

**X-MAGICAL**

Run the following bash script to download the demonstration dataset for the X-MAGICAL benchmark:

```bash
bash scripts/download_xmagical_dataset.sh
```

The dataset will be located in `/tmp/xirl/datasets/xmagical`.

**EXPERIMENTS**

The previous bash script should have downloaded a saved_models folder with our experiments with their checkpoint files. If there are any issues, please download it directly here: https://drive.google.com/file/d/1MWxzIwvqzXEHsr0R7hbqZchNHM6sZYnR/view


## Code Navigation

At a high-level, our code relies on two important but generic python scripts: `pretrain.py` for pretraining and `train_policy.py` for reinforcement learning. We use [ml_collections](https://github.com/google/ml_collections) to parameterize these scripts with experiment-specific config files. **All experiments must use config files that inherit from the base config files in `base_configs/`**. Specifically, pretraining experiments must inherit from `base_configs/pretrain.py` and RL experiments must inherit from `base_configs/rl.py`.

The rest of the codebase is organized as follows:

* `configs/` contains all config files used in our CoRL submission. They inherit from `base_configs/`.
* `xirl/` is the core pretraining codebase.
* `sac/` is the core Soft-Actor-Critic implementation adapted from [pytorch_sac](https://github.com/denisyarats/pytorch_sac).
* `scripts/` contains miscellaneous bash scripts.

## Experiments

**Same-embodiment setting**

To pretrain our model using TCC loss with a specific embodiment: `python pretrain_xmagical_same_embodiment.py --algo xirl --embodiment shortstick`
Then to use SAC reinforcement learning to better the model: `python rl_xmagical_learned_reward.py --pretrained_path ./saved_models/dataset=xmagical_mode=same_algo=xirl_embodiment=shortstick`
To view one of our experiments, navigate to the folder:
`env_name=SweepToTop-Longstick-State-Allo-TestLayout-v0_reward=learned_reward_type=distance_to_goal_mode=same_algo_algo=xirl_uid=60ba5eb6-22a4-4c7f-9e45-3d06fa92655e`
Note that each subdirectory is a different seed with a folder `./video/eval` with the agent learning the task with its corresponding embodiment.

**Cross-embodiment setting**

To pretrain our model using TCC loss using different embodiments than specified for cross-embodiment training: `python pretrain_xmagical_cross_embodiment.py --algo xirl --embodiment shortstick`
Then to use SAC reinforcement learning to better the model: `python pretrain_xmagical_cross_embodiment.py --pretrained_path ./saved_models/dataset=xmagical_mode=cross_algo=xirl_embodiment=gripper`
To view one of our experiments, navigate to the folder:
`env_name=SweepToTop-Gripper-State-Allo-TestLayout-v0_reward=learned_reward_type=distance_to_goal_mode=cross_algo=xirl_uid=8d844e1c-3b10-42be-9ea5-b01d456ddfa0`
Note that each subdirectory is a different seed with a folder `./video/eval` with the agent learning the task with its corresponding embodiment.


**Interactive reward visualization**

To interact with the agent as well as generate graphs, use the following script: `python interact_reward.py`

## Acknowledgments

Our work was based off the research below: 
`@inproceedings{zakka2021xirl,
  author    = {Zakka, Kevin and Zeng, Andy and Florence, Pete and Tompson, Jonathan and Bohg, Jeannette and Dwibedi, Debidatta},
  title     = {XIRL: Cross-embodiment Inverse Reinforcement Learning},
  booktitle = {Proceedings of the 5th Conference on Robot Learning (CoRL)},
  year      = {2021},
}`
