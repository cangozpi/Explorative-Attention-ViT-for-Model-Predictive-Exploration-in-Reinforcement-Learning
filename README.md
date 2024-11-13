# Explorative Attention ViT for Model-Predictive Exploration in Reinforcement Learning

## VIT with Explorative Attention

<img src="./assets/architecture overview.png" width="100%" title="Overview of the ViT with Explorative Attention approach.">

Figure: Overview of the ViT with Explorative Attention approach.

<img src="./assets/intuitive visualization.png" width="70%" title="Intutitive visualization of the Explorative Attention Mechanism">

Figure: An intuitive visualization of how the Explorative Attention Module
should ideally work is shown. The image on the left demonstrates that exploitative
attention should pay attention to the skulls. The image in the middle shows that
explorative attention should pay attention to the ladder. The image on the right
shows the outcome of aggregating the explorative and exploitative attention features.

## Motivation:
Inspired by the success of transformers in vision and sequential decision making
tasks, we aimed to harness their strengths for exploration problems in reinforcement
learning. Our goal was to improve learned representations by learning two different
representations specific to exploration and exploitation. We believed that successful
implementation of this idea could lead to learning better representations through
architectural changes alone, without the need for SSL. This approach can be viewed
as representation learning through architectural changes for prediction-error based
exploration methods in reinforcement learning. Among available transformer mod-
els for vision, we chose to build upon the Vision Transformer (ViT) architecture
([Dosovitskiy et al., 2021]) due to its widespread use in research and the availability
of its code implementation.

We wanted to modify the ViT architecture so that it could learn where in the input
images to pay attention for exploration and where to focus for exploitation. For in-
stance, the explorative attention (exploration related attention) should learn to pay
attention to exploration related features in a given input image, such as unexplored
doors and ladders in the Montezuma’s Revenge environment, as these can lead to
the exploration of new rooms, indicative of the exploratory behaviour. On the other
hand, the exploitative attention (exploitation related attention) should learn to pay
attention to exploitation related features in a given input image, such as skulls and
lava on the floor in the Montezuma’s Revenge environment, as these can harm the
agent and impact the extrinsic rewards (refer to the Figure).

We wanted our proposed architecture to use the extracted explorative features for
predicting intrinsic rewards using PPO’s intrinsic value head, and to use the ex-
ploitative features for predicting extrinsic rewards using PPO’s extrinsic value head.
Both sets of extracted features (extracted explorative features and the extracted
exploitative features) are aggregated (we used element-wise summation) to predict
the next action using PPO’s policy head. We aggregate the features because we
want the policy to consider both exploration related features and exploitation re-
lated features when making a decision.

The explorative features are extracted by appending a learnable exploration token
in front of the sequence of image patches and using the exploration token’s query
vector to obtain the exploration features. Similarly, the exploitative features are
extracted by appending a learnable exploitation token in front of the sequence of
image patches and using the exploitation token’s query vector to obtain the exploita-
tion features. This proposed ViT architecture, which we called Vit with Explorative
Attention, serves as a backbone that replaces the convolutional PPO backbone in
the RND approach. We tested this idea with RND, but it could be adapted for
other methods that provide intrinsic rewards.

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
## Running:
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

    See the available SLURM job scrips under _configs/*_.

---
* Continue Training RND agent from a checkpoint in Montezuma's Revenge:
    1. set _loadModel = True_ in the corresponding config file.
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=32 main.py --train --num_env_per_process 64 --config_path=./configs/expGlados3/Montezuma/config_originalRND_NoSSL_VitExplorativeAttnLucidrains.conf --log_name=montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed42_expGlados3-2_cont00 --save_model_path=checkpoints/expGlados3/Montezuma/montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed42_expGlados3-2_cont00.ckpt --seed=42 --gpu_id=1 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
    ```
* Note that this examples uses 1 node and 64 processes in total. It will have 1 process as agent/trainer and another 64 processes as environment workers. You can modify the parameters of torchrun to suit your needs.

---
* Test a trained agent in Montezuma's Revenge:
    1. set _loadModel = True_ in the corresponding config file.
    2. use _--eval_ command line argument when running the code.
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=34 main.py --eval --num_env_per_process 1 --config_path=./configs/expGlados3/Montezuma/config_originalRND_NoSSL_VitExplorativeAttnLucidrains.conf --log_name=montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed43_expGlados3-4_eval --load_model_path=checkpoints/expGlados3/Montezuma/montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed43_expGlados3-4.ckpt --seed=42 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
    ```

---
### Profiling
* Profiling with Scalene (torchrun Example):
    ```bash
    python -m scalene --- -m torch.distributed.run --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=32 main.py --train --num_env_per_process 64 --config_path=./configs/expGlados3/Montezuma/config_originalRND_NoSSL_VitExplorativeAttnLucidrains.conf --log_name=montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed42_expGlados3-2_cont00 --save_model_path=checkpoints/expGlados3/Montezuma/montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed42_expGlados3-2_cont00.ckpt --seed=42 --gpu_id=1 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8
    ```

* Profiling with Pytorch Profiler (torchrun Example):
    ```bash
    torchrun --nnodes 1 --nproc_per_node 1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --rdzv-id=32 main.py --train --num_env_per_process 64 --config_path=./configs/expGlados3/Montezuma/config_originalRND_NoSSL_VitExplorativeAttnLucidrains.conf --log_name=montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed42_expGlados3-2_cont00 --save_model_path=checkpoints/expGlados3/Montezuma/montezuma_originalRND_NoSSL_VitExplorativeAttnLucidrains_seed42_expGlados3-2_cont00.ckpt --seed=42 --gpu_id=1 --use_wandb --wandb_api_key=d012c9698bf568b1807b1cfe9ed56611311573e8 --pytorch_profiling
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

### Supported OpenAI GYM Environments
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


### Appendix:
__Model Predictive Exploration:__
* Random Network Distillation (RND): 
    
    * Paper: https://arxiv.org/abs/1810.12894
    * Code: https://github.com/jcwleo/random-network-distillation-pytorch

    ---

__Transformer:__
* An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale:

    * Paper: https://arxiv.org/abs/2010.11929
    * Code: https://github.com/lucidrains/vit-pytorch

