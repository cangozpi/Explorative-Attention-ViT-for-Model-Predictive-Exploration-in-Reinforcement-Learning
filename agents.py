import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel
from utils import global_grad_norm_
from utils import Logger
from config import default_config

from torch import nn

import torch.profiler
from torch.profiler import ProfilerActivity
from dist_utils import get_dist_info
from config import args
from model import ViT_IMPLEMENTATION
from math import radians

from utils import Env_action_space_type
from torch.distributions.normal import Normal


import wandb

class RNDAgent(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            env_action_space_type,
            num_env,
            num_step,
            gamma,
            GAE_Lambda=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            max_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            representation_lr_method="BYOL",
            device = None,
            logger:Logger=None):
        super().__init__()
        self.env_action_space_type = env_action_space_type

        if int(default_config['ViT_implementation_type']) == ViT_IMPLEMENTATION.LUCIDRAINS_ViT.value:
            ViT_implementation_type = ViT_IMPLEMENTATION.LUCIDRAINS_ViT
        elif int(default_config['ViT_implementation_type']) == ViT_IMPLEMENTATION.HG_ViT.value:
            ViT_implementation_type = ViT_IMPLEMENTATION.HG_ViT
        else:
            ViT_implementation_type = None

        self.model = CnnActorCriticNetwork(input_size, output_size, self.env_action_space_type, use_noisy_net, ViT_implementation_type=ViT_implementation_type)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.GAE_Lambda = GAE_Lambda
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.max_grad_norm = max_grad_norm
        self.update_proportion = update_proportion
        # self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.use_cuda = use_cuda
        if use_cuda:
            self.device = device
        else:
            self.device = 'cpu'
        assert isinstance(logger, Logger)
        self.logger = logger

        extracted_feature_embedding_dim = self.model.extracted_feature_embedding_dim
        self.train_method = default_config['TrainMethod']
        assert self.train_method in ['PPO', 'original_RND', 'modified_RND']
        self.rnd = None
        if self.train_method == 'original_RND':
            self.rnd = RNDModel(input_size=input_size, output_size=512, train_method=self.train_method)
        elif self.train_method == 'modified_RND':
            self.rnd = RNDModel(input_size=extracted_feature_embedding_dim, output_size=512, train_method=self.train_method)
        
        assert representation_lr_method in ['None', "BYOL", "Barlow-Twins"]
        self.representation_lr_method = representation_lr_method
        self.representation_model = None
        self.representation_loss_coef = 0
        # --------------------------------------------------------------------------------
        # for BYOL (Bootstrap Your Own Latent)
        if self.representation_lr_method == "BYOL":
            backbone_model = self.model.feature
            self.backbone_model = backbone_model
            from BYOL import BYOL, Augment
            BYOL_projection_hidden_size = int(default_config['BYOL_projectionHiddenSize'])
            BYOL_projection_size = int(default_config['BYOL_projectionSize'])
            BYOL_moving_average_decay = float(default_config['BYOL_movingAverageDecay'])
            apply_same_transform_to_batch = default_config.getboolean('apply_same_transform_to_batch')
            self.representation_model = BYOL(backbone_model, in_features=extracted_feature_embedding_dim, projection_size=BYOL_projection_size, projection_hidden_size=BYOL_projection_hidden_size, moving_average_decay=BYOL_moving_average_decay, batch_norm_mlp=True, use_cuda=use_cuda, device=device) # Model used to perform representation learning (e.g. BYOL)
            self.data_transform = Augment(input_size, apply_same_transform_to_batch=apply_same_transform_to_batch)
            self.representation_loss_coef = float(default_config['BYOL_representationLossCoef'])
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # for Barlow-Twins
        if self.representation_lr_method == "Barlow-Twins":
            backbone_model = self.model.feature
            self.backbone_model = backbone_model
            from BarlowTwins import BarlowTwins, Transform
            import json
            projection_sizes = json.loads(default_config['BarlowTwinsProjectionSizes'])
            BarlowTwinsLambda = float(default_config['BarlowTwinsLambda'])
            apply_same_transform_to_batch = default_config.getboolean('apply_same_transform_to_batch')
            self.representation_model = BarlowTwins(backbone_model, in_features=extracted_feature_embedding_dim, projection_sizes=projection_sizes, lambd=BarlowTwinsLambda, use_cuda=use_cuda, device=device) # Model used to perform representation learning (e.g. BYOL)
            self.data_transform = Transform(input_size, apply_same_transform_to_batch=apply_same_transform_to_batch)
            self.representation_loss_coef = float(default_config['BarlowTwins_representationLossCoef'])
        # --------------------------------------------------------------------------------

        self.optimizer = optim.Adam(self.get_agent_parameters(), lr=learning_rate)

        self.model = self.model.to(self.device)
        if self.rnd is not None:
            self.rnd = self.rnd.to(self.device)
        if self.representation_model is not None:
            self.representation_model = self.representation_model.to(self.device)
        

        self.freeze_shared_backbone_during_training = default_config.getboolean('freeze_shared_backbone')
        
    
    def get_agent_parameters(self):
        """
        Gathers the parameters (nn.Module parameters) of the RNDAgent and returns the unique ones 
        (without repetition of the shared parameters btw different models/modules).
        In other words returns parameters from PPO Agent, RND, and Representation Model (i.e. BYOL, Barlow-Twins).
        """
        agent_params = list(self.model.parameters()) # PPO params
        if self.rnd is not None: # RND params
            agent_params = [*agent_params, *list(self.rnd.predictor.parameters())]
        if self.representation_model is not None: # representation model params (i.e. BYOL | Barlow-Twins)
            agent_params = [*agent_params, *list(self.representation_model.get_trainable_parameters())]
        agent_params = set(agent_params)
        
        # double checking
        for p in self.model.parameters():
            assert p in agent_params
        if self.rnd is not None:
            for p in self.rnd.predictor.parameters():
                assert p in agent_params
        if self.representation_model is not None:
            for p in self.representation_model.get_trainable_parameters():
                assert p in agent_params

        return agent_params


    def set_mode(self, mode="train"):
        """
        Sets torch Modules (models) of the agent to the specified mode.
        """
        assert mode in ["train", "eval"]
        if mode == "train":
            self.model = self.model.train()
            if self.rnd is not None:
                self.rnd = self.rnd.train()
            if self.representation_model is not None:
                self.representation_model = self.representation_model.train()
        elif mode == "eval":
            self.model = self.model.eval()
            if self.rnd is not None:
                self.rnd = self.rnd.eval()
            if self.representation_model is not None:
                self.representation_model = self.representation_model.eval()
        else:
            raise NotImplementedError()

    def get_action(self, state):
        state = torch.Tensor(state).type(torch.float32).to(self.device)
        if self.env_action_space_type == Env_action_space_type.DISCRETE:
            policy, value_ext, value_int = self.model(state)
            action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

            action = self.random_choice_prob_index(action_prob)

            return action, value_ext.detach().cpu().numpy().squeeze(), value_int.detach().cpu().numpy().squeeze(), policy.detach().cpu().numpy()

        elif self.env_action_space_type == Env_action_space_type.CONTINUOUS:
            mu, std, value_ext, value_int = self.model(state)
            cont_dist = Normal(mu, std)
            action = cont_dist.sample()
            logp_a = cont_dist.log_prob(action).sum(axis=-1).unsqueeze(-1)

            return action.detach().cpu().numpy(), value_ext.detach().cpu().numpy().squeeze(), value_int.detach().cpu().numpy().squeeze(), logp_a.detach().cpu().numpy()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        assert (self.rnd is not None) and (self.train_method != 'PPO'), 'RND cannot be used when using "TrainMethod" = "PPO"'
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).mean(1)

        return intrinsic_reward.data.cpu().numpy()
    
    def extract_feature_embeddings(self, states):
        """
        Extracts feature embedding using the backbone model of PPO. Note that this backbone (feature extractor) model is
        what is used as the shared feature extractor in our proposed approach.
        """
        states = torch.FloatTensor(states).to(self.device)

        # Feature extraction used for old CNN PPO backbone:
        # extracted_feature_embeddings = self.model.feature(states)
        # return extracted_feature_embeddings

        if self.model.use_explorativeAttn: # Modified ViT with Explorative Attention using exploration_token and exploitation_token
            if self.model.ViT_implementation_type == ViT_IMPLEMENTATION.LUCIDRAINS_ViT:
                # Explorative Attention:
                x_explorative = self.model.feature(state, attn_type=ViT_Attn.EXPLORATIVE_ATTN) # [num_env * num_step, ViT_dim]

                # Exploitative Attention:
                x_exploitative = self.model.feature(state, attn_type=ViT_Attn.EXPLOITATIVE_ATTN) # [num_env * num_step, ViT_dim]

            elif self.model.ViT_implementation_type == ViT_IMPLEMENTATION.HG_ViT:
                # Explorative and Exploitative Attention:
                exploration_BaseModelOutputWithPooling, exploitation_BaseModelOutputWithPooling = self.model.feature(state) # [num_env * num_step, ViT_dim]

                # Explorative Attention:
                x_explorative = exploration_BaseModelOutputWithPooling[0][:, 0, :] # extract exploration_token's embeddings -> [num_env * num_step, ViT_dim]

                # Exploitative Attention:
                x_exploitative = exploitation_BaseModelOutputWithPooling[0][:, 0, :] # extract exploitation_token's embeddings -> [num_env * num_step, ViT_dim]

            return x_explorative, x_exploitative
        
        else: # Regular ViT attention with CLS Token
            if self.model.ViT_implementation_type == ViT_IMPLEMENTATION.LUCIDRAINS_ViT:
                x_cls = self.model.feature(state, attn_type=ViT_Attn.CLS_ATTN) # [num_env * num_step, ViT_dim]

            elif self.model.ViT_implementation_type == ViT_IMPLEMENTATION.HG_ViT:
                cls_BaseModelOutputWithPooling = self.model.feature(state) # [num_env * num_step, ViT_dim]
                x_cls = cls_BaseModelOutputWithPooling[0][:, 0, :] # extract CLS TOKEN's embeddings -> [num_env * num_step, ViT_dim]

            return x_cls



    def train_model(self, states, target_ext, target_int, y, adv, normalized_extracted_feature_embeddings, old_policy, global_update):
        _, GLOBAL_RANK, _, _ = get_dist_info()
        pytorch_profiler_log_path = f'./logs/torch_profiler_logs/{args["log_name"]}_agentTrainModel_prof_rank{GLOBAL_RANK}.log'
        self.logger.create_new_pytorch_profiler(pytorch_profiler_log_path, 1, 1, 3, 1)



        sample_range = np.arange(len(states))
        forward_mse = nn.MSELoss(reduction='none')

            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)

            total_loss, total_actor_loss, total_critic_loss, total_critic_loss_int, total_critic_loss_ext, total_entropy_loss, total_rnd_loss, total_representation_loss, \
                total_approx_kl, total_max_kl, total_entropy, total_clipfrac = [], [], [], [], [], [], [], [], [], [], [], []
            total_grad_norm_unclipped = []
            if default_config.getboolean('UseGradClipping'):
                total_grad_norm_clipped = []

            for j in range(int(len(states) / self.batch_size)):
                batch_indices = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # Perform batching and sending to GPU:
                s_batch = torch.FloatTensor(states)[batch_indices].to(self.device)
                target_ext_batch = torch.FloatTensor(target_ext)[batch_indices].to(self.device)
                if self.train_method in ['original_RND', 'modified_RND']:
                    target_int_batch = torch.FloatTensor(target_int)[batch_indices].to(self.device)
                if self.env_action_space_type == Env_action_space_type.DISCRETE:
                    y_batch = torch.LongTensor(y)[batch_indices].to(self.device)
                elif self.env_action_space_type == Env_action_space_type.CONTINUOUS:
                    y_batch = torch.FloatTensor(y)[batch_indices].to(self.device)
                adv_batch = torch.FloatTensor(adv)[batch_indices].to(self.device)
                if self.train_method in ['original_RND', 'modified_RND']:
                    normalized_extracted_feature_embeddings_batch = torch.FloatTensor(normalized_extracted_feature_embeddings)[batch_indices].to(self.device)
                with torch.no_grad():
                    if self.env_action_space_type == Env_action_space_type.DISCRETE:
                        policy_old_list = torch.tensor(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size)[batch_indices].to(self.device) # --> [num_env*batch_size, output_size]
                        m_old = Categorical(F.softmax(policy_old_list, dim=-1))
                        log_prob_old = m_old.log_prob(y_batch)

                        if False: # for debugging
                            policy1 = self.model(torch.FloatTensor(states).to(self.device))[0].contiguous().view(-1, self.output_size)[batch_indices].to(self.device) # This should equal to 'policy_old_list'
                            assert torch.allclose(policy_old_list, policy1), "Something is wrong with the indexing of old_policy and y_batch correspondance !"

                    elif self.env_action_space_type == Env_action_space_type.CONTINUOUS:
                        log_prob_old = torch.tensor(old_policy).permute(1, 0, 2).contiguous().view(-1, 1)[batch_indices].to(self.device) # --> [num_env*batch_size, 1] # Note: old_policy corresponds to 'logp_a' values for CONTINUOUS action space

                        if False: # for debugging
                            mu1, std1, _, _  = self.model(torch.FloatTensor(states).to(self.device)) # [num_env*batch_size, 1]
                            mu1 = mu1.contiguous().view(-1, self.output_size)[batch_indices].to(self.device) # [batch_size, 1]
                            cont_dist1 = torch.distributions.normal.Normal(mu1, std1)
                            logp_a1 = cont_dist1.log_prob(y_batch.unsqueeze(-1)).sum(axis=-1).unsqueeze(-1) # This should be equal to log_prob_old
                            assert torch.allclose(log_prob_old, logp_a1), "Something is wrong with the indexing of states and y_batch correspondance !"

                            # another check of the smae thing which directly uses s_batch:
                            mu1, std1, _, _  = self.model(s_batch) # [num_env*batch_size, 1]
                            mu1 = mu1.contiguous().view(-1, self.output_size) # [batch_size, 1]
                            cont_dist1 = torch.distributions.normal.Normal(mu1, std1)
                            logp_a1 = cont_dist1.log_prob(y_batch.unsqueeze(-1)).sum(axis=-1).unsqueeze(-1) # This should be equal to log_prob_old
                            assert torch.allclose(log_prob_old, logp_a1), "Something is wrong with the indexing of states and y_batch correspondance !"



                rnd_loss = 0
                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                if self.rnd is not None:
                    # Note that gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                    predict_next_state_feature, target_next_state_feature = self.rnd(normalized_extracted_feature_embeddings_batch)
                    rnd_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                    # Proportion of exp used for predictor update
                    mask = torch.rand(len(rnd_loss)).to(self.device)
                    mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                    rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                    # Diagnostic metrics to log:
                    with torch.no_grad():
                        rnd_target_features_batch_dim_variance = torch.mean(torch.var(target_next_state_feature.detach(), dim=0)).cpu().numpy()
                        rnd_target_features_feat_dim_variance = torch.mean(torch.var(target_next_state_feature.detach(), dim=1)).cpu().numpy()
                        rnd_target_features_mean = torch.mean(target_next_state_feature.detach()).cpu().numpy()
                        rnd_target_features_max = torch.max(torch.abs(target_next_state_feature.detach())).cpu().numpy()
                        rnd_pred_features_batch_dim_variance = torch.mean(torch.var(predict_next_state_feature.detach(), dim=0)).cpu().numpy()
                        rnd_pred_features_feat_dim_variance = torch.mean(torch.var(predict_next_state_feature.detach(), dim=1)).cpu().numpy()
                        rnd_pred_features_mean = torch.mean(predict_next_state_feature.detach()).cpu().numpy()
                        rnd_pred_features_max = torch.max(torch.abs(predict_next_state_feature.detach())).cpu().numpy()
                        rnd_input_flattened = torch.reshape(normalized_extracted_feature_embeddings_batch, (normalized_extracted_feature_embeddings_batch.shape[0], -1)).detach() # [B, *]
                        rnd_input_batch_dim_variance = torch.mean(torch.var(rnd_input_flattened, dim=0)).cpu().numpy()
                        rnd_input_feat_dim_variance = torch.mean(torch.var(rnd_input_flattened, dim=1)).cpu().numpy()
                        rnd_input_mean = torch.mean(rnd_input_flattened).cpu().numpy()
                        rnd_input_max = torch.max(torch.abs(rnd_input_flattened)).cpu().numpy()
                # ---------------------------------------------------------------------------------


                representation_loss = 0
                # --------------------------------------------------------------------------------
                # for BYOL (Bootstrap Your Own Latent):
                if (self.representation_lr_method == "BYOL") and (self.freeze_shared_backbone_during_training == False):
                    # sample image transformations and transform the images to obtain the 2 views
                    B, STATE_STACK_SIZE, H, W = s_batch.shape
                    with torch.no_grad():
                        if default_config.getboolean('apply_same_transform_to_batch'):
                            s_batch_views = self.data_transform(torch.reshape(s_batch, [-1, H, W])[:, None, :, :]) # -> [B*STATE_STACK_SIZE, C=1, H, W], [B*STATE_STACK_SIZE, C=1, H, W]
                        else:
                            s_batch_views = self.data_transform(s_batch) # -> [B, C=STATE_STACK_SIZE, H, W], [B, C=STATE_STACK_SIZE, H, W]
                    s_batch_view1, s_batch_view2 = torch.reshape(s_batch_views[0], [B, STATE_STACK_SIZE, H, W]), \
                        torch.reshape(s_batch_views[1], [B, STATE_STACK_SIZE, H, W]) # -> [B, STATE_STACK_SIZE, H, W], [B, STATE_STACK_SIZE, H, W]
                
                    assert self.representation_model.net is self.model.feature # make sure that BYOL net and RL algo's feature extractor both point to the same network

                    # plot original frame vs transformed views for debugging purposes
                    if False:
                        import matplotlib.pyplot as plt
                        for i in range(4):
                            idx = np.random.choice(B)
                            print(idx)
                            fig, axs = plt.subplots(4, 2, constrained_layout=True)
                            axs[0,0].imshow(s_batch[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[0,1].imshow(s_batch_view1[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,0].imshow(s_batch[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,1].imshow(s_batch_view1[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,0].imshow(s_batch[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,1].imshow(s_batch_view1[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,0].imshow(s_batch[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,1].imshow(s_batch_view1[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')

                            axs[0,0].set_title(f'original state:{idx} frame:0', fontsize=10)
                            axs[0,1].set_title(f'augmented state:{idx} frame:0', fontsize=10)
                            axs[1,0].set_title(f'original state:{idx} frame:1', fontsize=10)
                            axs[1,1].set_title(f'augmented state:{idx} frame:1', fontsize=10)
                            axs[2,0].set_title(f'original state:{idx} frame:2', fontsize=10)
                            axs[2,1].set_title(f'augmented state:{idx} frame:2', fontsize=10)
                            axs[3,0].set_title(f'original state:{idx} frame:3', fontsize=10)
                            axs[3,1].set_title(f'augmented state:{idx} frame:3', fontsize=10)
                            plt.show()

                    # compute BYOL loss
                    BYOL_loss = self.representation_model(s_batch_view1, s_batch_view2) 
                    representation_loss = BYOL_loss
                # ---------------------------------------------------------------------------------


                # --------------------------------------------------------------------------------
                # for Barlow-Twins:
                if (self.representation_lr_method == "Barlow-Twins") and (self.freeze_shared_backbone_during_training == False):
                    # sample image transformations and transform the images to obtain the 2 views
                    B, STATE_STACK_SIZE, H, W = s_batch.shape
                    with torch.no_grad():
                        if default_config.getboolean('apply_same_transform_to_batch'):
                            s_batch_views = self.data_transform(torch.reshape(s_batch, [-1, H, W])[:, None, :, :]) # -> [B*STATE_STACK_SIZE, C=1, H, W], [B*STATE_STACK_SIZE, C=1, H, W]
                        else:
                            s_batch_views = self.data_transform(s_batch) # -> [B, C=STATE_STACK_SIZE, H, W], [B, C=STATE_STACK_SIZE, H, W]
                    s_batch_view1, s_batch_view2 = torch.reshape(s_batch_views[0], [B, STATE_STACK_SIZE, H, W]), \
                        torch.reshape(s_batch_views[1], [B, STATE_STACK_SIZE, H, W]) # -> [B, STATE_STACK_SIZE, H, W], [B, STATE_STACK_SIZE, H, W]
                
                    assert self.representation_model.backbone is self.model.feature # make sure that Barlow-Twins backbone and RL algo's feature extractor both point to the same network

                    # plot original frame vs transformed views for debugging purposes
                    if False:
                        import matplotlib.pyplot as plt
                        for i in range(4):
                            idx = np.random.choice(B)
                            print(idx)
                            fig, axs = plt.subplots(4, 2, constrained_layout=True)
                            axs[0,0].imshow(s_batch[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[0,1].imshow(s_batch_view1[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,0].imshow(s_batch[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,1].imshow(s_batch_view1[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,0].imshow(s_batch[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,1].imshow(s_batch_view1[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,0].imshow(s_batch[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,1].imshow(s_batch_view1[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')

                            axs[0,0].set_title(f'original state:{idx} frame:0', fontsize=10)
                            axs[0,1].set_title(f'augmented state:{idx} frame:0', fontsize=10)
                            axs[1,0].set_title(f'original state:{idx} frame:1', fontsize=10)
                            axs[1,1].set_title(f'augmented state:{idx} frame:1', fontsize=10)
                            axs[2,0].set_title(f'original state:{idx} frame:2', fontsize=10)
                            axs[2,1].set_title(f'augmented state:{idx} frame:2', fontsize=10)
                            axs[3,0].set_title(f'original state:{idx} frame:3', fontsize=10)
                            axs[3,1].set_title(f'augmented state:{idx} frame:3', fontsize=10)
                            plt.show()

                    # compute Barlow-Twins loss
                    BarlowTwins_loss = self.representation_model(s_batch_view1, s_batch_view2) 
                    representation_loss = BarlowTwins_loss
                # ---------------------------------------------------------------------------------


                # --------------------------------------------------------------------------------
                # for Proximal Policy Optimization (PPO):
                if self.env_action_space_type == Env_action_space_type.DISCRETE:
                    policy, value_ext, value_int = self.model(s_batch)
                    m = Categorical(F.softmax(policy, dim=-1))
                    log_prob = m.log_prob(y_batch)

                elif self.env_action_space_type == Env_action_space_type.CONTINUOUS:
                    mu, std, value_ext, value_int  = self.model(s_batch)
                    mu = mu.contiguous().view(-1, self.output_size) # [batch_size, 1]
                    m = torch.distributions.normal.Normal(mu, std)
                    log_prob = m.log_prob(y_batch.unsqueeze(-1)).sum(axis=-1).unsqueeze(-1)


                ratio = torch.exp(log_prob - log_prob_old) # Note that for the first pass (i.e. epoch=0) ratio has to equal 1

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch

                actor_loss = -torch.min(surr1, surr2).mean()

                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch)
                critic_int_loss = 0
                if self.rnd is not None:
                    critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch)

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean() # note that this is used both for loss calculation and logging
                # PPO's diagnostic logging metrics (refer to: https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/ppo/ppo.html):
                with torch.no_grad():
                    approx_kl = (log_prob_old - log_prob).detach().mean().item()
                    max_kl = (log_prob_old - log_prob).detach().max().item()
                    clipped = ratio.gt(1+self.ppo_eps) | ratio.lt(1-self.ppo_eps)
                    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                # --------------------------------------------------------------------------------

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + rnd_loss + self.representation_loss_coef * representation_loss
                loss.backward()

                grad_norm_unclipped = global_grad_norm_(self.get_agent_parameters())
                if default_config.getboolean('UseGradClipping'):
                    nn.utils.clip_grad_norm_(self.get_agent_parameters(), self.max_grad_norm) # gradient clipping
                    grad_norm_clipped = global_grad_norm_(self.get_agent_parameters())
                # Log final model grads in detail
                if i == self.epoch - 1 and default_config.getboolean('verbose_logging') == True:
                    self.logger.log_gradients_in_model_to_tb_without_step(self.model,'PPO', log_full_detail=True, only_rank_0=True)
                    if self.rnd is not None:
                        self.logger.log_gradients_in_model_to_tb_without_step(self.rnd, 'RND', log_full_detail=True, only_rank_0=True)
                    if self.representation_model is not None:
                        self.logger.log_gradients_in_model_to_tb_without_step(self.representation_model, f'{self.representation_lr_method}', log_full_detail=True, only_rank_0=True)

                self.optimizer.step()

                # logging
                total_loss.append(loss.detach().cpu().item())
                total_actor_loss.append(actor_loss.detach().cpu().item())
                total_critic_loss.append(0.5 * critic_loss.detach().cpu().item())
                if self.rnd is not None:
                    total_critic_loss_int.append(0.5 * critic_int_loss.detach().cpu().item())
                total_critic_loss_ext.append(0.5 * critic_ext_loss.detach().cpu().item())
                total_entropy_loss.append(- self.ent_coef * entropy.detach().cpu().item())
                total_entropy.append(entropy.detach().cpu().item())
                total_approx_kl.append(approx_kl)
                total_max_kl.append(max_kl)
                total_clipfrac.append(clipfrac)
                if self.rnd is not None:
                    total_rnd_loss.append(rnd_loss.detach().cpu().item())
                if (self.representation_model is not None) and (self.freeze_shared_backbone_during_training == False):
                    total_representation_loss.append(self.representation_loss_coef * representation_loss.detach().cpu().item())
                total_grad_norm_unclipped.append(grad_norm_unclipped)
                if default_config.getboolean('UseGradClipping') == True:
                    total_grad_norm_clipped.append(grad_norm_clipped)

                # EMA update BYOL target network params
                if self.representation_lr_method == "BYOL":
                    self.representation_model.update_moving_average()


                self.logger.step_pytorch_profiler(pytorch_profiler_log_path)


            # logging (wandb):
            if self.logger.use_wandb and GLOBAL_RANK == 0:
                epoch_log_dict = {
                    'train/overall_loss (everything combined) vs epoch': np.mean(total_loss),
                    'train/PPO_actor_loss vs epoch': np.mean(total_actor_loss),
                    'train/PPO_critic_loss (intrinsic + extrinsic) vs epoch': np.mean(total_critic_loss),
                    'train/PPO_critic_loss (extrinsic) vs epoch': np.mean(total_critic_loss_ext),
                    'train/PPO_entropy_loss vs epoch': np.mean(total_entropy_loss),
                    'train/PPO entropy vs epoch': np.mean(total_entropy),
                    'train/PPO approximate_KL vs epoch': np.mean(total_approx_kl),
                    'train/PPO max_KL vs epoch': np.max(total_max_kl),
                    'train/PPO clipfrac vs epoch': np.mean(total_clipfrac),
                    'train/grad_norm_unclipped vs epoch': np.mean(total_grad_norm_unclipped),
                }
                if self.rnd is not None:
                    epoch_log_dict = {
                        **epoch_log_dict,
                        'train/PPO_critic_loss (intrtinsic) vs epoch': np.mean(total_critic_loss_int),
                        'train/RND_loss vs epoch': np.mean(total_rnd_loss),
                        'train/rnd_target_features_batch_dim_variance vs epoch': rnd_target_features_batch_dim_variance,
                        'train/rnd_target_features_feat_dim_variance vs epoch': rnd_target_features_feat_dim_variance,
                        'train/rnd_target_features_mean vs epoch': rnd_target_features_mean,
                        'train/rnd_target_features_max vs epoch': rnd_target_features_max,
                        'train/rnd_pred_features_batch_dim_variance vs epoch': rnd_pred_features_batch_dim_variance,
                        'train/rnd_pred_features_feat_dim_variance vs epoch': rnd_pred_features_feat_dim_variance,
                        'train/rnd_pred_features_mean vs epoch': rnd_pred_features_mean,
                        'train/rnd_pred_features_max vs epoch': rnd_pred_features_max,
                        'train/rnd_input_batch_dim_variance vs epoch': rnd_input_batch_dim_variance,
                        'train/rnd_input_feat_dim_variance vs epoch': rnd_input_feat_dim_variance,
                        'train/rnd_input_mean vs epoch': rnd_input_mean,
                        'train/rnd_input_max vs epoch': rnd_input_max,
                    }
                if self.representation_model is not None:
                    epoch_log_dict = {
                        **epoch_log_dict,
                        f'train/Representation_loss({self.representation_lr_method}) vs epoch': np.mean(total_representation_loss),
                    }
                if default_config.getboolean('UseGradClipping') == True:
                    epoch_log_dict = {
                        **epoch_log_dict,
                        'train/grad_norm_clipped vs epoch': np.mean(total_grad_norm_clipped),
                        }
                epoch_log_dict = {f'wandb_{k}': v for (k, v) in epoch_log_dict.items()}
                epoch = self.logger.tb_global_steps['epoch']
                wandb.log({
                    'epoch': epoch,
                    **epoch_log_dict
                })
                self.logger.tb_global_steps['epoch'] = self.logger.tb_global_steps['epoch'] + 1

            # logging (tb):
            self.logger.log_scalar_to_tb_without_step('train/overall_loss (everything combined) vs epoch', np.mean(total_loss), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO_actor_loss vs epoch', np.mean(total_actor_loss), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO_critic_loss (intrinsic + extrinsic) vs epoch', np.mean(total_critic_loss), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO_critic_loss (extrinsic) vs epoch', np.mean(total_critic_loss_ext), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO_entropy_loss vs epoch', np.mean(total_entropy_loss), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO entropy vs epoch', np.mean(total_entropy), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO approximate_KL vs epoch', np.mean(total_approx_kl), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO max_KL vs epoch', np.max(total_max_kl), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/PPO clipfrac vs epoch', np.mean(total_clipfrac), only_rank_0=True)
            self.logger.log_scalar_to_tb_without_step('train/grad_norm_unclipped vs epoch', np.mean(total_grad_norm_unclipped), only_rank_0=True)
            if self.rnd is not None:
                self.logger.log_scalar_to_tb_without_step('train/PPO_critic_loss (intrtinsic) vs epoch', np.mean(total_critic_loss_int), only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/RND_loss vs epoch', np.mean(total_rnd_loss), only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_target_features_batch_dim_variance vs epoch', rnd_target_features_batch_dim_variance, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_target_features_feat_dim_variance vs epoch', rnd_target_features_feat_dim_variance, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_target_features_mean vs epoch', rnd_target_features_mean, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_target_features_max vs epoch', rnd_target_features_max, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_pred_features_batch_dim_variance vs epoch', rnd_pred_features_batch_dim_variance, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_pred_features_feat_dim_variance vs epoch', rnd_pred_features_feat_dim_variance, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_pred_features_mean vs epoch', rnd_pred_features_mean, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_pred_features_max vs epoch', rnd_pred_features_max, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_input_batch_dim_variance vs epoch', rnd_input_batch_dim_variance, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_input_feat_dim_variance vs epoch', rnd_input_feat_dim_variance, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_input_mean vs epoch', rnd_input_mean, only_rank_0=True)
                self.logger.log_scalar_to_tb_without_step('train/rnd_input_max vs epoch', rnd_input_max, only_rank_0=True)
            if self.representation_model is not None:
                self.logger.log_scalar_to_tb_without_step(f'train/Representation_loss({self.representation_lr_method}) vs epoch', np.mean(total_representation_loss), only_rank_0=True)
            if default_config.getboolean('UseGradClipping') == True:
                self.logger.log_scalar_to_tb_without_step('train/grad_norm_clipped vs epoch', np.mean(total_grad_norm_clipped), only_rank_0=True)
            if default_config.getboolean('verbose_logging') == True:
                # Log final model parameters in detail
                self.logger.log_parameters_in_model_to_tb_without_step(self.model, f'PPO', only_rank_0=True)
                if self.rnd is not None:
                    self.logger.log_parameters_in_model_to_tb_without_step(self.rnd, f'RND', only_rank_0=True)
                if self.representation_model is not None:
                    self.logger.log_parameters_in_model_to_tb_without_step(self.representation_model, f'{self.representation_lr_method}', only_rank_0=True)
    

    def add_tb_graph(self, batch_size, stateStackSize, input_size):
        """
        Logs a graph of forward passes of RNN, PPO, and Representation Model used under tensorboard's graph tab.
        """
        _, GLOBAL_RANK, _, _ = get_dist_info()
        if GLOBAL_RANK == 0:
            with torch.no_grad():
                def graph_forward(dummy_state_batch):
                    return_vals = []
                    if self.model is not None: # PPO
                        if self.model.env_action_space_type == Env_action_space_type.DISCRETE:
                            policy, value_ext, value_int = self.model(dummy_state_batch)
                            return_vals += [policy, value_ext, value_int]
                        elif self.model.env_action_space_type == Env_action_space_type.CONTINUOUS:
                            mu, std, value_ext, value_int = self.model(dummy_state_batch)
                            return_vals += [mu, std, value_ext, value_int]
                    if self.representation_model is not None: # Representation model
                        representation_output = self.representation_model(dummy_state_batch, dummy_state_batch)
                        return_vals += [representation_output]
                    if self.rnd is not None: # RND
                        if self.train_method == 'original_RND':
                            extracted_feature_embeddings = dummy_state_batch[:, -1, None, :, :]
                        elif self.train_method == 'modified_RND':
                            extracted_feature_embeddings = self.extract_feature_embeddings(dummy_state_batch.detach().cpu().numpy() / 255).to(self.device) # [(num_step * num_env_workers), feature_embeddings_dim]
                        predict_next_state_feature, target_next_state_feature = self.rnd(extracted_feature_embeddings)
                        return_vals += [predict_next_state_feature, target_next_state_feature]
                    
                    # return policy, value_ext, value_int, representation_output, predict_next_state_feature, target_next_state_feature
                    return tuple(return_vals)

                tmp_forward = self.forward
                self.forward = graph_forward
                dummy_state_batch = torch.zeros(batch_size, stateStackSize, input_size, input_size).to(self.device)
                self.logger.tb_summaryWriter.add_graph(self, dummy_state_batch)
                self.forward = tmp_forward
