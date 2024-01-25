import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from config import default_config
from vit import ViT_Attn, ViT
from transformers import ViTConfig
from vit_hg import ViT_ExplorativeAttn
from enum import Enum


class ViT_IMPLEMENTATION(Enum):
    LUCIDRAINS_ViT = 0
    HG_ViT = 1


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    # Refer to: https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py for the architecture

    # self.extracted_feature_embedding_dim  = 448
    extracted_feature_embedding_dim  = int(default_config['extracted_feature_embedding_dim']) # TODO: make this corerspond to ViT's extracted feature embedding dim so that modified_rnd and SSL parts of the code can be used again

    def __init__(self, input_size, output_size, use_noisy_net=False, ViT_implementation_type: ViT_IMPLEMENTATION=ViT_IMPLEMENTATION.LUCIDRAINS_ViT):
        super(CnnActorCriticNetwork, self).__init__()
        self.ViT_implementation_type = ViT_implementation_type
        assert isinstance(ViT_implementation_type, ViT_IMPLEMENTATION), 'ViT_implementation_type must be of type enum ViT_IMPLEMENTATION'

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        # --------------- original paper's architecture below:
        # self.feature = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=4, # TODO: this equals StateStackSize
        #         out_channels=32,
        #         kernel_size=8,
        #         stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=32,
        #         out_channels=64,
        #         kernel_size=4,
        #         stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=64,
        #         out_channels=64,
        #         kernel_size=3,
        #         stride=1),
        #     nn.ReLU(),
        #     Flatten(),
        #     linear(
        #         7 * 7 * 64,
        #         256),
        #     nn.ReLU(),
        #     linear(
        #         256,
        #         CnnActorCriticNetwork.extracted_feature_embedding_dim), # = 448 # TODO: set extracted_feature_embedding_dim to 448 to match original RND paper !
        #     nn.ReLU()
        # )

        # self.actor = nn.Sequential(
        #     linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, CnnActorCriticNetwork.extracted_feature_embedding_dim),
        #     nn.ReLU(),
        #     linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, output_size)
        # )

        # self.extra_layer = nn.Sequential(
        #     linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, CnnActorCriticNetwork.extracted_feature_embedding_dim),
        #     nn.ReLU()
        # )

        # self.critic_ext = linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, 1)
        # self.critic_int = linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, 1)

        # for p in self.modules():
        #     if isinstance(p, nn.Conv2d):
        #         init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        #     if isinstance(p, nn.Linear):
        #         init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        # init.orthogonal_(self.critic_ext.weight, 0.01)
        # self.critic_ext.bias.data.zero_()

        # init.orthogonal_(self.critic_int.weight, 0.01)
        # self.critic_int.bias.data.zero_()

        # for i in range(len(self.actor)):
        #     if type(self.actor[i]) == nn.Linear:
        #         init.orthogonal_(self.actor[i].weight, 0.01)
        #         self.actor[i].bias.data.zero_()

        # for i in range(len(self.extra_layer)):
        #     if type(self.extra_layer[i]) == nn.Linear:
        #         init.orthogonal_(self.extra_layer[i].weight, 0.1)
        #         self.extra_layer[i].bias.data.zero_()

        

        # --------------- Visition Transformer (lucidrains Pytorch Implementation) architecture below:
        if self.ViT_implementation_type == ViT_IMPLEMENTATION.LUCIDRAINS_ViT:
            ViT_dim = int(default_config['ViTlucidrains_dim'])
            self.feature = ViT(image_size=int(default_config['PreProcHeight']),
                                patch_size=int(default_config['ViTlucidrains_patch_size']), # 12, TODO: set this from config
                                num_classes=int(default_config['ViTlucidrains_num_classes']), # -1, no classification head
                                dim=int(default_config['ViTlucidrains_dim']), # equals 16*64 = heads*dim_head = 1024
                                depth=int(default_config['ViTlucidrains_depth']), # 6
                                heads=int(default_config['ViTlucidrains_heads']), # 16
                                mlp_dim=int(default_config['ViTlucidrains_mlp_dim']), # 2048
                                dropout=float(default_config['ViTlucidrains_dropout']), # 0.1
                                emb_dropout=float(default_config['ViTlucidrains_emb_dropout']), # 0.1
                                channels=int(default_config['StateStackSize']), # for regular ViT this is 3 (RGB), but for our use its 4 (StateStackSize)
                                dim_head=int(default_config['ViTlucidrains_dim_head']), # 64
                                use_explorativeAttn=default_config.getboolean('ViTlucidrains_use_explorativeAttn')
                            )

        
        # --------------- Visition Transformer (Modified HuggingFace Implementation) architecture below:
        elif self.ViT_implementation_type == ViT_IMPLEMENTATION.HG_ViT:
            ViT_dim = int(default_config['ViTHG_hidden_size'])
            ViT_config = ViTConfig(
                hidden_size=int(default_config['ViTHG_hidden_size']), # originally it was 768
                num_hidden_layers=int(default_config['ViTHG_num_hidden_layers']), # originally it was 12
                num_attention_heads=int(default_config['ViTHG_num_attention_heads']), # originally it was 12
                intermediate_size=int(default_config['ViTHG_intermediate_size']), # 3072
                hidden_act="gelu",
                hidden_dropout_prob=float(default_config['ViTHG_hidden_dropout_prob']),
                attention_probs_dropout_prob=float(default_config['ViTHG_attention_probs_dropout_prob']),
                initializer_range=float(default_config['ViTHG_initializer_range']),
                layer_norm_eps=float(default_config['ViTHG_layer_norm_eps']),
                image_size=int(default_config['ViTHG_PreProcHeight']),
                patch_size=int(default_config['ViTHG_patch_size']),
                num_channels=int(default_config['ViTHG_StateStackSize']), # for regular ViT this is 3 (RGB), but for our use it is 4 (StateStackSize)
                qkv_bias=default_config.getboolean('ViTHG_qkv_bias'),
                encoder_stride=int(default_config['ViTHG_encoder_stride']),
            )
            self.feature = ViT_ExplorativeAttn(ViT_config, add_pooling_layer=True, use_mask_token=False, use_explorativeAttn=default_config.getboolean('ViTHG_use_explorativeAttn'))



        # --------------- Visition Transformer Heads Used by all of available ViT implementations:
        self.actor = nn.Sequential(
            linear(ViT_dim, ViT_dim),
            nn.ReLU(),
            linear(ViT_dim, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(ViT_dim, ViT_dim),
            nn.ReLU()
        )

        self.critic_ext = linear(ViT_dim, 1)
        self.critic_int = linear(ViT_dim, 1)
        

    def forward(self, state, attn_aggregation_op='mean'):
        """
        attn_aggregation_op (str): one of ['mean', 'sum']. Specifies the operation used to aggregate the features extracted by explorative attention
            and exploitative attention. The aggregated features are passed onto policy network to obtain an action distribution.
        """
        # state -> [num_env * num_step, StateStackSize, H, W]
        if self.ViT_implementation_type == ViT_IMPLEMENTATION.LUCIDRAINS_ViT:
            if self.feature.use_explorativeAttn: # Modified ViT's Explorative Attention with exploration_token and exploitation_token
                # Explorative Attention:
                x_explorative = self.feature(state, attn_type=ViT_Attn.EXPLORATIVE_ATTN) # [num_env * num_step, ViT_dim]
                value_int = self.critic_int(self.extra_layer(x_explorative) + x_explorative) # [num_env * num_step, 1]

                # Exploitative Attention:
                x_exploitative = self.feature(state, attn_type=ViT_Attn.EXPLOITATIVE_ATTN) # [num_env * num_step, ViT_dim]
                value_ext = self.critic_ext(self.extra_layer(x_exploitative) + x_exploitative) # [num_env * num_step, 1]

        
                # Explorative + Exploitative Attention:
                if attn_aggregation_op == 'mean':
                    x_combined = torch.concat((x_explorative.unsqueeze(dim=1), x_exploitative.unsqueeze(dim=1)), dim=1) # [num_env * num_step, 2, ViT_dim]
                    x_combined = torch.mean(x_combined, dim=1, keepdim=False) # [num_env * num_step, ViT_dim]
                elif attn_aggregation_op == 'sum':
                    x_combined = x_explorative + x_exploitative # aggregate feature by summation from both attention mechanisms -> [num_env * num_step, ViT_dim]
                else:
                    assert attn_aggregation_op in ['mean', 'sum'], 'attention_aggregation_op must be one of ["mean", "sum"]'

                policy = self.actor(x_combined) # [num_env * num_step, ViT_dim
            
            else: # Regular ViT attention with CLS Token
                # CLS Token Attention:
                x_cls = self.feature(state, attn_type=ViT_Attn.CLS_ATTN) # [num_env * num_step, ViT_dim]
                value_int = self.critic_int(self.extra_layer(x_cls) + x_cls) # [num_env * num_step, 1]
                value_ext = self.critic_ext(self.extra_layer(x_cls) + x_cls) # [num_env * num_step, 1]
                policy = self.actor(x_cls) # [num_env * num_step, ViT_dim
        

        elif self.ViT_implementation_type == ViT_IMPLEMENTATION.HG_ViT:
            if self.feature.use_explorativeAttn: # Modified ViT's Explorative Attention with exploration_token and exploitation_token
                # Explorative and Exploitative Attention:
                exploration_BaseModelOutputWithPooling, exploitation_BaseModelOutputWithPooling = self.feature(state) # [num_env * num_step, ViT_dim]

                # Explorative Attention:
                x_explorative = exploration_BaseModelOutputWithPooling[0][:, 0, :] # extract exploration_token's embeddings -> [num_env * num_step, ViT_dim]
                value_int = self.critic_int(self.extra_layer(x_explorative) + x_explorative) # [num_env * num_step, 1]

                # Exploitative Attention:
                x_exploitative = exploitation_BaseModelOutputWithPooling[0][:, 0, :] # extract exploitation_token's embeddings -> [num_env * num_step, ViT_dim]
                value_ext = self.critic_int(self.extra_layer(x_exploitative) + x_exploitative) # [num_env * num_step, 1]
        
                # Explorative + Exploitative Attention:
                if attn_aggregation_op == 'mean':
                    x_combined = torch.concat((x_explorative.unsqueeze(dim=1), x_exploitative.unsqueeze(dim=1)), dim=1) # [num_env * num_step, 2, ViT_dim]
                    x_combined = torch.mean(x_combined, dim=1, keepdim=False) # [num_env * num_step, ViT_dim]
                elif attn_aggregation_op == 'sum':
                    x_combined = x_explorative + x_exploitative # aggregate feature by summation from both attention mechanisms -> [num_env * num_step, ViT_dim]
                else:
                    assert attn_aggregation_op in ['mean', 'sum'], 'attention_aggregation_op must be one of ["mean", "sum"]'

                policy = self.actor(x_combined) # [num_env * num_step, ViT_dim

            else: # Regular ViT attention with CLS Token
                # CLS Token Attention:
                cls_BaseModelOutputWithPooling = self.feature(state) # [num_env * num_step, ViT_dim]
                x_cls = cls_BaseModelOutputWithPooling[0][:, 0, :] # extract CLS TOKEN's embeddings -> [num_env * num_step, ViT_dim]
                value_int = self.critic_int(self.extra_layer(x_cls) + x_cls) # [num_env * num_step, 1]
                value_ext = self.critic_int(self.extra_layer(x_cls) + x_cls) # [num_env * num_step, 1]
                policy = self.actor(x_cls) # [num_env * num_step, ViT_dim


        return policy, value_ext, value_int


class RNDModel(nn.Module):
    # Refer to: https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py for the architecture
    def __init__(self, input_size=32, output_size=512, train_method="modified_RND"):
        super(RNDModel, self).__init__()
        assert train_method in ['original_RND', 'modified_RND']

        self.input_size = input_size
        self.output_size = output_size

        if train_method == 'original_RND':
            feature_output = 7 * 7 * 64
            self.predictor = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )

            self.target = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, output_size)
            )

        elif train_method == 'modified_RND':
            self.predictor = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),

                nn.Linear(256, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )

            self.target = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),

                nn.Linear(256, output_size),
            )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


