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


class SiglipEncoder(nn.Module):
    pass


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super(SiglipVisionEmbeddings, self).__init__()
        self.config = config
        self.embed_dim = self.config["SiglipVisionConfig"]["hidden_size"]
        self.image_size = self.config["SiglipVisionConfig"]["image_size"]
        self.patch_size = self.config["SiglipVisionConfig"]["patch_size"]

        # feature extractor intput (the image 3 channels) , the output (embedding 768) with kernel of 16 (patch size) 
        self.patch_embedding = nn.Conv2d(
            in_channels = self.config["SiglipVisionConfig"]['num_channels'],
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size,
            padding = 'valid'
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions , self.embed_dim)
        self.register_buffer(
            'positions_ids',
            torch.arrange(self.num_positions).expand((1,-1)),
            presistent = False,
        )
    
    def forward(self , inputs: torch.FloatTensor) -> torch.Tensor:
        _ , _ , H , W = inputs.shape  #B , C , H , W
        patch_embeds = self.patch_embedding(inputs) #



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


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super(SiglipVisionModel, self).__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, inputs) -> Tuple:
        # Patchify the input images. [Batch size , channels , H , W] -> [B , Num_patches , latent_size]
        self.vision_model(inputs)
