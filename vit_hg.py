# Modifies ViT model from HuggingFace with exploration and exploitation specific attention (i.e. queries).

# For ViT model HG documentation see: https://huggingface.co/docs/transformers/model_doc/vit#transformers.ViTModel
# For torch implementaiton of HG ViT model see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py

from transformers import AutoImageProcessor, ViTModel, ViTConfig, PreTrainedModel 
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTPatchEmbeddings, ViTEncoder, ViTPooler, BaseModelOutputWithPooling
import torch
from torch import nn
from typing import Dict, List, Optional, Set, Tuple, Union


# ------------------------------------------------
# Loading a pretrained model from HG:
# image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
# model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# inputs = image_processor(image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# ------------------------------------------------

# ViT_config = ViTConfig(
#     hidden_size=768,
#     num_hidden_layers=12,
#     num_attention_heads=12,
#     intermediate_size=3072,
#     hidden_act="gelu",
#     hidden_dropout_prob=0.0,
#     attention_probs_dropout_prob=0.0,
#     initializer_range=0.02,
#     layer_norm_eps=1e-12,
#     image_size=224,
#     patch_size=12,
#     num_channels=4, # for regular ViT this is 3 (RGB), we use 4 (StateStackSize),
#     qkv_bias=True,
#     encoder_stride=16
# )

# vit_model = ViTModel(ViT_config, add_pooling_layer=True, use_mask_token=False)


class ViTEmbeddings_ExplorativeAttn(nn.Module):
    """
    Modifies forward function of VitEmbeddings to use exploration_token and exploitation_token instead of cls_token.
    For the original HG İmplementation refer to: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py
    """
    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
            super().__init__()

            # Remove cls_token and add exploration_token and exploitative_token (modification 1):
            # self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) # regular ViT uses this, we use exploration_token and exploitation_token
            self.exploration_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) # Exploration Token
            self.exploitation_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) # Exploitation Token

            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
            self.patch_embeddings = ViTPatchEmbeddings(config)
            num_patches = self.patch_embeddings.num_patches
            self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.config = config


    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0] # Note that in our case (modified ViT with explorative attention), this still holds even though it is not class_pos anymore
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask



        # ----- APPLY EXPLORATIVE ATTENTION:
        # add the [exploration_token] to the embedded patch tokens
        exploration_tokens = self.exploration_token.expand(batch_size, -1, -1)
        exploration_embeddings = torch.cat((exploration_tokens, embeddings), dim=1)

        # add positional encoding to each exploration token
        if interpolate_pos_encoding:
            exploration_embeddings = exploration_embeddings + self.interpolate_pos_encoding(exploration_embeddings, height, width)
        else:
            exploration_embeddings = exploration_embeddings + self.position_embeddings

        exploration_embeddings = self.dropout(exploration_embeddings)
        

        # ----- APPLY EXPLOITATIVE ATTENTION:
        # add the [exploitation_token] to the embedded patch tokens
        exploitation_tokens = self.exploitation_token.expand(batch_size, -1, -1)
        exploitation_embeddings = torch.cat((exploitation_tokens, embeddings), dim=1)

        # add positional encoding to each exploration token
        if interpolate_pos_encoding:
            exploitation_embeddings = exploitation_embeddings + self.interpolate_pos_encoding(exploitation_embeddings, height, width)
        else:
            exploitation_embeddings = exploitation_embeddings + self.position_embeddings


        exploitation_embeddings = self.dropout(exploitation_embeddings)


        # ----- APPLY ViT ATTENTION with CLS Token:
        # add the [CLS] token to the embedded patch tokens
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        # if interpolate_pos_encoding:
        #     embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        # else:
        #     embeddings = embeddings + self.position_embeddings

        # embeddings = self.dropout(embeddings)

        # return embeddings

        return exploration_embeddings, exploitation_embeddings


    
class ViTPreTrainedModel_ExplorativeAttn(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit" #TODO: might have to chage this to "vit_explorativeAttn"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "ViTLayer", "ViTEmbeddings_ExplorativeAttn"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings): # Regular ViT with CLS TOKEN uses this
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            # Initialize CLS Token for regular ViT
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
        elif isinstance(module, ViTEmbeddings_ExplorativeAttn): # Modified ViT with exploration_token and exploitation_token uses this
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            # Inıtialize Exploration Token for ViT_ExplorativeAttn
            module.exploration_token.data = nn.init.trunc_normal_(
                module.exploration_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.exploration_token.dtype)

            # Inıtialize Exploitation Token for ViT_ExploitativeAttn
            module.exploitation_token.data = nn.init.trunc_normal_(
                module.exploitation_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.exploitation_token.dtype)


class ViT_ExplorativeAttn(ViTPreTrainedModel_ExplorativeAttn):
    """
    Modifies ViT model from HuggingFace with exploration and exploitation specific attention (i.e. queries).
    Modifications made:
        1. support exploration_token and exploitation token instead of cls_token
        2. initialization should handle removal of cls_token with exploration_token and exploitation_token
        3. forward methods should be able to use explorative attentation and exploitative attention
        4. save&load functionality should support saving and loading of exploration_token and exploitation_token
    """
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False, use_explorativeAttn:bool=True):
        super().__init__(config)
        self.use_explorativeAttn = use_explorativeAttn # If True use Modified ViT with ExplorativeAttention, else use regular ViT implementation with CLS token
        self.config = config

        # Remove cls_token and add exploration_token and exploitative_token (modification 1):
        if self.use_explorativeAttn:
            self.embeddings = ViTEmbeddings_ExplorativeAttn(config, use_mask_token=use_mask_token) # For Modified ViT which uses exploration_token and exploitation_token
        else:
            self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token) # For Regular ViT with CLS TOKEN

        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings


    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    # @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPooling,
    #     config_class=_CONFIG_FOR_DOC,
    #     modality="vision",
    #     expected_output=_EXPECTED_OUTPUT_SHAPE,
    # )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)



        if self.use_explorativeAttn: # ---- For Modified ViT with Explorative Attention:
            # ---- Get Explorative Attention (Token) Embeddings (for ViT_ExplorativeAttn):
            explorationEmbedding_output, exploitationEmbedding_output = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
            )


            # ---- Apply Explorative Attention (for ViT_ExplorativeAttn):
            exploration_encoder_outputs = self.encoder(
                explorationEmbedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            exploration_sequence_output = exploration_encoder_outputs[0]
            exploration_sequence_output = self.layernorm(exploration_sequence_output)
            exploration_pooled_output = self.pooler(exploration_sequence_output) if self.pooler is not None else None

            if not return_dict: # TODO: this and the one below should not return ! fix this !!! ASAP
                exploration_head_outputs = (exploration_sequence_output, exploration_pooled_output) if exploration_pooled_output is not None else (exploration_sequence_output,)
                exploration_outputs = exploration_head_outputs + exploration_encoder_outputs[1:]

            exploration_BaseModelOutputWithPooling = BaseModelOutputWithPooling(
                last_hidden_state=exploration_sequence_output,
                pooler_output=exploration_pooled_output,
                hidden_states=exploration_encoder_outputs.hidden_states,
                attentions=exploration_encoder_outputs.attentions,
            )
    

            # ---- Apply Exploitative Attention (for ViT_ExplorativeAttn):
            exploitation_encoder_outputs = self.encoder(
                exploitationEmbedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            exploitation_sequence_output = exploitation_encoder_outputs[0]
            exploitation_sequence_output = self.layernorm(exploitation_sequence_output)
            exploitation_pooled_output = self.pooler(exploitation_sequence_output) if self.pooler is not None else None

            if not return_dict:
                exploitation_head_outputs = (exploitation_sequence_output, exploitation_pooled_output) if exploitation_pooled_output is not None else (exploitation_sequence_output,)
                exploitation_outputs = exploitation_head_outputs + exploitation_encoder_outputs[1:]

            exploitation_BaseModelOutputWithPooling = BaseModelOutputWithPooling(
                last_hidden_state=exploitation_sequence_output,
                pooler_output=exploitation_pooled_output,
                hidden_states=exploitation_encoder_outputs.hidden_states,
                attentions=exploitation_encoder_outputs.attentions,
            )



            if not return_dict:
                return exploration_outputs, exploitation_outputs


            return exploration_BaseModelOutputWithPooling, exploitation_BaseModelOutputWithPooling


        else: # ---- For Regular ViT with CLS Token:
            embedding_output = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
            )

            encoder_outputs = self.encoder(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]
            sequence_output = self.layernorm(sequence_output)
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

            if not return_dict:
                head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
                return head_outputs + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
