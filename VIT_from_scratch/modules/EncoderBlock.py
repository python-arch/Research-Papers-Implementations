import torch.nn as nn
import torch

class EncoderBlock(nn.Module):
    def __init__(self, model_config):
        super(EncoderBlock , self).__init__()
        self.latent_size = model_config['params']['latent_size']
        self.num_heads = model_config['params']['num_head']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dropout = model_config['params']['dropout']
        
        # Normalization layer
        self.norm = nn.LayerNorm(self.latent_size)

        self.multihead = nn.MultiheadAttention(
            self.latent_size , self.num_heads , dropout = self.dropout
        )

        # here is mlp is Linear + GELU
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size , self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4 , self.latent_size),
            nn.Dropout(self.dropout)
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
    
    def forward(self, embedded_patches):
        f_norm_out = self.norm(embedded_patches)
        # output is (attn_output , attn_output_weights)
        attention_out = self.multihead(f_norm_out , f_norm_out , f_norm_out)[0]

        # first residual connection
        first_added = embedded_patches + attention_out

        s_norm_out = self.norm(first_added)
        ff_out = self.enc_MLP(s_norm_out)

        return ff_out + first_added # the second resdiual connectin in the encoder block



