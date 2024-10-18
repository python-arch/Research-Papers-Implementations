import torch
import torch.nn as nn
from typing import Optional, Tuple
import yaml

with open(
    "/home/abdelrahman.elsayed/implementations/PaliGemma/modules/SigLip/siglip_config.yaml",
    "r",
) as f:
    config = yaml.safe_load(f)

print(config)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super(SiglipVisionEmbeddings, self).__init__()
        self.config = config
        self.embed_dim = self.config["SiglipVisionConfig"]["hidden_size"]
        self.image_size = self.config["SiglipVisionConfig"]["image_size"]
        self.patch_size = self.config["SiglipVisionConfig"]["patch_size"]

        # feature extractor intput (the image 3 channels) , the output (embedding 768) with kernel of 16 (patch size)
        # input -> [B , 3 , H , W]
        self.patch_embedding = nn.Conv2d(
            in_channels=self.config["SiglipVisionConfig"]["num_channels"],
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        # output -> [B , 768  , H_2 , W_2] // H_2 = H / patch_size , // W_2 = W / patch_size

        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2  # assuming square image
        self.num_positions = (
            self.num_patches
        )  # for each patch we need positional_encoding vector
        self.position_embedding = nn.Embedding(
            self.num_positions, self.embed_dim
        )  # [num_patches , 768]
        self.register_buffer(
            "positions_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            presistent=False,
        )

    def forward(self, inputs: torch.FloatTensor) -> torch.Tensor:
        _, _, H, W = inputs.shape  # B , C , H , W
        patch_embeds = self.patch_embedding(
            inputs
        )  # apply convolutions to get [ B , Embed_dim , Num_patches_H , num_pathces_W]
        # we need to flatten the embed
        embeddings = patch_embeds.flatten(2)  # to get [B , Embed_Dim , num_patches]
        embeddings = embeddings.transpose(1, 2)  # to get [B , num_patches , embed_dim]
        embeddings = embeddings + self.position_embedding(
            self.position_ids
        )  # to add positional encoding
        # the final embeddings are of shape [B, num_patches , 768] + position embeddings [1, position_id , 768] for each positon
        return embeddings


class SigLipAttention(nn.Module):
    def __init__(self, config):
        super(SigLipAttention, self).__init__()
        self.config = config
        self.embed_dim = config["SiglipVisionConfig"]["hidden_size"]
        self.num_heads = config["SiglipVisionConfig"]["num_attention_heads"]
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = (
            self.head_dim**-0.5
        )  # in the formula for the attention 1/sqrt(head_dim)
        self.dropout = config["SiglipVisionConfig"]["attention_dropout"]

        # Key , Query , Value
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # inputs -> [B , Num_patches , Embed_dim]
        batch_size, seq_len, _ = inputs.size()
        # Query states
        query_states = self.q_proj(inputs)
        # Key states
        key_states = self.k_proj(inputs)
        # Value states
        value_states = self.v_proj(inputs)
        # split the query states  , we have multi heads each head is responsible of relating the tokens in different ways
        # [B , Num_patches , embed_dim ] -> [B , num_patches , num_heads , head_dim] -> [B , num_heads , num_patches , head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # in the vision part the query and key and value are the same number of heads
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # why transpose?
        # for parralelization , for each head it can access indepedntly all the patches at the same time from different embeddings

        # Calculate Attention Weights using Q*K.T / sqrt(d_k). attn_weights = [B , num_heads , num_patches , num_patches]?
        # [B , num_heads , num_patches , head_dim] * [B , num_heads , head_dim , num_patches]
        # This represents the dot products for each token and other tokens for each head
        # for each head we capture the relationship between the patch and all other patches

        # why scaling? Numerical stablitiy , stable gradients, softmax is sensitive to large input values
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        )
        # ASSERT the dimensions
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of shape {(batch_size , self.num_heads , seq_len , seq_len)} , but the found shape is",
                f"{attn_weights.size()}",
            )
        # apply softmax
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # we don't use dropout but added it to be consistent with hugging face implementation
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        # next , we multiply the attention weights by the value -> [B , NNum_heads , num_patches , head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        # we need to assert the size
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_output should be of size {(batch_size , self.num_heads , seq_len , self.head_dim)} , but is"
                f"{attn_output.size()}"
            )
        # transpose attn_output to get [B , num_heads , num_patches , head_dim] -> [B , num_patches , num_heads , head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [B , Num_patches , num_heads , head_dim ] -> [B , num_patches , embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # last thing , we multiply by Wo..
        # why? because the calculation for each attention score is independent for each head , so the contextualization is independent also
        # we basically we need to mix all of this not just concatentating it
        # [B , Num_patches , num_heads , head_dim ] -> same shape
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super(SiglipMLP, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(
            config["SiglipVisionConfig"]["hidden_size"],
            config["SiglipVisionConfig"]["intermediate_size"],
        )
        self.fc2 = nn.Linear(
            config["SiglipVisionConfig"]["intermediate_size"],
            config["SiglipVisionConfig"]["hidden_size"],
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.fc1(inputs)
        inputs = nn.functional.gelu(inputs, approximate="tanh")
        inputs = self.fc2(inputs)
        return inputs


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SiglipEncoderLayer, self).__init__()
        self.config = config
        self.embed_dim = config["SiglipVisionConfig"]["hidden_size"]
        self.selt_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(
            self.embed_dim, eps=config["SiglipVisionConfig"]["layer_norm_eps"]
        )
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(
            self.embed_dim, eps=config["SiglipVisionConfig"]["layer_norm_eps"]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # resdiual : [B , num_patches , embed_dim]
        resdiual = inputs  # aka hidden states
        inputs = self.layer_norm1(inputs)  # Normalize (shape is the same)
        inputs, _ = self.selt_attn(inputs)  # self attention (shape is the same)
        inputs = inputs + resdiual  # skip connection
        resdiual = inputs  # saved for the second skip connection
        inputs = self.layer_norm2(inputs)  # normalize again
        inputs = self.mlp(
            inputs
        )  # more degrees of freedom , non-linearity , transforms the features to be ready for the next layers
        inputs = resdiual + inputs
        return inputs


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super(SiglipEncoder, self).__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                SiglipEncoderLayer(config)
                for _ in range(self.config["SiglipVisionConfig"]["num_hidden_layers"])
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs [B , num_patches , embed_dim]
        hidden_states = inputs
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states
            )  # [B , num_patches , embed_dim] -> [B , num_patches , embed_dim]
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super(SiglipVisionTransformer, self).__init__()
        self.config = config
        self.embed_dim = self.config["SiglipVisionConfig"]["hidden_size"]
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            self.embed_dim, eps=config["SiglipVisionConfig"]["layer_norm_eps"]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # embeddings
        hidden_states = self.embeddings(inputs)
        # encode them
        output_embedddings = self.encoder(input_embeds=hidden_states)
        output_embedddings = self.post_layernorm(output_embedddings)

        # Normalize and return the output embeddings
        return output_embedddings


"""
This is a wrapper class for the Vision model
"""


# in the contrastive learning we said we are representing the image with one embedding , however the output from the vision transformer is [B , num_ptches , embed_dim]
# What do we do?
# we take the first patch as representative for the whole model or average the embeddings over the patches axis


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super(SiglipVisionModel, self).__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, inputs) -> Tuple:
        # Patchify the input images. [Batch size , channels , H , W] -> [B , Num_patches , latent_size]
        self.vision_model(inputs)
