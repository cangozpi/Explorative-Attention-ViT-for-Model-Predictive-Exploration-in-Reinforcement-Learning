# Explorative Attention ViT for Model-Predictive Exploration in Reinforcement Learning

## Motivation:
We introduce ViT with explorative attention mechanism. It learns intrinsic reward (exploration) related representations from an _exploration_token_ and extrinsic reward (exploitation) related representations from an _exploitation_token_. We test train this modified ViT model (i.e. ViT_ExplorativeAttn) on OpenAI GYM's MontezumaRevenge environment which is a sparse reward, hard exploration problem. Extrinsic rewards are returned by the environment, and the intrinsic rewards are obtained from Random Network Distillation (RND). Following from the RND paper we use Proximal Policy Optimization (PPO) to train the reinforcement learning agent.

---

### Installation:
_Note: developed using python==3.8.16, pip==23.0.1, ubuntu==22.04.3_
* Installation with Docker:
    ```bash
    make docker_build # create Image
    make docker_start # create and run Container
    ```
    ---

* Installation with conda:
    ```bash
    conda create --name <env> python=3.8.16 --file requirements.txt
    ```
## Usage:

* Training RND only:
    Use a config file with _RepresentationLearningParameter = None_.
* Train RND + BYOL:
    Use a config file with _RepresentationLearningParameter = BYOL_.
* Train RND + Barlow-Twins:
    Use a config file with _RepresentationLearningParameter = Barlow-Twins_.


---

### Torchrun Distributed Training/Testing (Example):

* Train RND agent in MontezumaRevenge from scratch:
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/MontezumaRevenge/config_rnd00.conf --log_name=MontezumaRevenge_rnd00 --save_model_path=checkpoints/MontezumaRevenge/rnd00.ckpt
    ```

* Continue Training RND agent from a checkpoint in MontezumaRevenge:
    1. set _loadModel = True_ in the corresponding config file.
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --train --num_env_per_process 64 --config_path=./configs/MontezumaRevenge/config_rnd00.conf --log_name=MontezumaRevenge_rnd00_cont00 --save_model_path=checkpoints/MontezumaRevenge/rnd00_cont00.ckpt --load_model_path=checkpoints/MontezumaRevenge/rnd00.ckpt
    ```
* Note that this examples uses 1 node and 128 processes in total. It will have 1 process as agent/trainer and another 127 processes as environment workers. You can modify the parameters of torchrun to suit your needs.

---
* Test a RND agent in MontezumaRevenge:
    ```bash
    torchrun --nnodes 1 --nproc_per_node 2 --standalone main.py --eval --config_path=./configs/demo_config.conf --log_name=MontezumaRevenge_rnd00 --load_model_path=checkpoints/rnd00.ckpt
    ```
* Note that nproc_per_node has to be 2 for testing of the agent. This is because it supports only a single environment worker during training.

---
### Profiling
* Profiling with Scalene (Example):
    ```bash
    scalene --no-browser --cpu --gpu --memory --outfile profile_rnd_montezuma.html --profile-interval 120 main.py --train --config_path=./configs/demo_config.conf --log_name=rnd00 --save_model_path=checkpoints/rnd00.ckpt

    ```
* Profiling with Scalene (torchrun Example):
    ```bash
    python -m scalene --- -m torch.distributed.run --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt
    ```

* Profiling with Pytorch Profiler (torchrun Example):
    ```bash
    torchrun --nnodes 1 --nproc_per_node 3 --standalone main.py --train --config_path=./configs/demo_config.conf --log_name=demo_00 --save_model_path=checkpoints/demo_00.ckpt --pytorch_profiling
    ```
---

### Some helper commands
* Kill RND code (and its subprocesses):
    ```bash
    make kill
    ```
* Start Tensorboard Server:
    ```bash
    make start_tensorboard
    ```

---

## Overview
* __Configurations__:
    * Most of the running parameters/options are set inside a config file (_*.conf_). These files are located in _./configs_ directory. An explanation of the available options can be found by running:
        ```bash
        python3 main.py --train --config_options
        ```
    * Once you have a config file, you need to provide command line arguments to specify some other options. An explanation of the available command line options can be found by running:
        ```bash
        python3 main.py -h
        ```
    ---

* __Supported OpenAI GYM Environments__
    * Atari (https://www.gymlibrary.dev/environments/atari/index.html):
        * Montezuma Revenge:
            in the config file set:
            ```config
            EnvType = atari
            EnvID = MontezumaRevengeNoFrameskip-v4
            ```
        * Pong:
            in the config file set:
            ```config
            EnvType = atari
            EnvID = PongNoFrameskip-v4
            ```

    * Super Mario Bros (https://pypi.org/project/gym-super-mario-bros/):
        * Super Mario Bros:
            in the config file set:
            ```config
            EnvType = mario
            EnvID = SuperMarioBros-v0
            ```

    * Classic Control (https://www.gymlibrary.dev/environments/classic_control/):
        * Cart Pole:
            in the config file set:
            ```config
            EnvType = classic_control
            EnvID = CartPole-v1
            ```
    ---
- __Distributed Training Architecture__
    * The code relies on _torch.distributed_ package to implement distributed training. It is implemented so that every node is assigned a single agent (GPU) which gathers rollouts by interacting with the environment workers (CPU) and trains the agent. The rest of the processes in a given node are assigned as the environment workers. These processes have an instance of the gym environment and are used solely to interact with these environments in a parallelized manner. Every agent(trainer) process sends sends actions to the environment worker processes in their node and gathers interactions with the environments. Then, these interactions are used to train the model. Gradients across agent workers are synchronized by making use of the _DistributedDataParallel_ module of _torch_.
    
    * In every node, 1 process (process with local_rank == 0) is assigned to the agents_group, the remaining processes are
    assigned to the env_workers_group. To get a better understanding check out the example below.
    agents_group processes have an instance of RNDAgent and perform optimizations.
    env_workers_group processes have an instance of the environment and perform interactions with it.
        ```txt
        Example:

            Available from torchrun:
                nnodes: number of nodes = 3
                nproc_per_node: number of processes per node = 4
            ---

            ************** NODE 0:
            LOCAL_RANK 0: GPUs --> agents_group
            LOCAL_RANK != 0: CPUs --> env_workers_group
            **************
            ...

            ************** NODE: 1:
            LOCAL_RANK 0: GPUs --> agents_group
            LOCAL_RANK != 0: CPUs --> env_workers_group
            **************
            ...

            ************** NODE: 2:
            LOCAL_RANK 0: GPUs --> agents_group
            LOCAL_RANK != 0: CPUs --> env_workers_group
            **************

            -node0-  -node1-   -node2-
            0,1,2,3  4,5,6,7  8,9,10,11    ||    agents_group_ranks=[0,4,8], env_workers_group_rank=[remaining ranks]
            *        *        *
        ```
    ---

* __Tests__
    * _tests.py_: This file contains some tests for environment wrappers and custom environment implementations.

---


### Appendix:
__Model Predictive Exploration:__
* Random Network Distillation (RND): 
    
    * Paper: https://arxiv.org/abs/1810.12894
    * Code: https://github.com/jcwleo/random-network-distillation-pytorch

    ---

__Non-Contrastive Representation Learning:__
* BYOL-Explore: Exploration by Bootstrapped Prediction:

    * Paper: https://arxiv.org/abs/2206.08332


* Bootstrap Your Own Latent a New Approach to Self-Supervised Learning (BYOL):

    * Paper: https://arxiv.org/pdf/2006.07733.pdf
    * Code: 
        
        1. https://github.com/The-AI-Summer/byol-cifar10/blob/main/ai_summer_byol_in_cifar10.py#L92
        2. https://github.com/SaeedShurrab/Simple-BYOL/blob/master/byol.py
        3. https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

* Barlow Twins: Self-Supervised Learning via Redundancy Reduction

    * Paper: https://arxiv.org/pdf/2103.03230.pdf
    * Code:

        1. https://github.com/facebookresearch/barlowtwins
        2. https://github.com/MaxLikesMath/Barlow-Twins-Pytorch
