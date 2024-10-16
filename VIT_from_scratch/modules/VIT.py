import torch.nn as nn
import torch
from InputEmbedding import *
from EncoderBlock import *

class VIT(nn.Module):
    def __init__(self , model_config):
        super(VIT , self).__init__()
        self.num_encoders = model_config['params']['num_encoders']
        self.latent_size = model_config['params']['latent_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = model_config['params']['num_classes']
        self.dropout = model_config['params']['dropout']

        # get the embeddeing 
        self.embeddeing = InputEmbedding(model_config)
        # encoders blocks
        self.encoders = nn.ModuleList([EncoderBlock(model_config) for i in range(self.num_encoders)])
        # final mlphead
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.LayerNorm(self.latent_size , self.latent_size),
            nn.Linear(self.latent_size , self.num_classes)
        )
    
    def debug_tensor(self,tensor, name="Tensor"):
        """
        Prints debugging information for a PyTorch tensor.
        
        Args:
            tensor (torch.Tensor): The tensor to debug.
            name (str): The name of the tensor (for identification).
        """
        print(f"--- {name} Debug Info ---")
        print(f"Shape: {tensor.shape}")
        print(f"Data Type: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Requires Grad: {tensor.requires_grad}")
        print(f"Values: {tensor}")
        print(f"--- End of {name} Debug Info ---\n")

    def forward(self , test_input):
        enco_output = self.embeddeing(test_input)

        for encoder in self.encoders:
            enco_output = encoder(enco_output)
        
        # take the class token
        cls_token_embed = enco_output[:,0]

        return self.MLP_head(cls_token_embed)
    

        