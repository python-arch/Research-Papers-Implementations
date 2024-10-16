import torch.nn as nn
import torch
import einops


class InputEmbedding(nn.Module):
    def __init__(self, model_config):
        super(InputEmbedding, self).__init__()
        self.patch_size = model_config["params"]["patch_size"]
        self.n_channels = model_config["params"]["n_channels"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.latent_size = model_config["params"]["latent_size"]
        self.batch_size = model_config["training"]["batch_size"]
        # just for test with one input
        # self.batch_size = 1
        self.input_size = self.n_channels * self.patch_size * self.patch_size

        # Linear Projection
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)
        # Class token
        self.class_token = nn.Parameter(
            torch.randn(self.batch_size, 1, self.latent_size)
        ).to(self.device)
        # Positional Embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(self.batch_size, 1, self.latent_size)
        ).to(self.device)

    def debug_tensor(self, tensor, name="Tensor"):
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

    def forward(self, input_data):
        input_data = input_data.to(self.device)

        # Patchify input images
        patches = einops.rearrange(
            input_data,
            "b c (h h1) (w w1) -> b (h w) (h1 w1 c)",
            h1=self.patch_size,
            w1=self.patch_size,
        )

        # print(patches.size())
        # print(input_data.size())

        # project them
        linear_projections = self.linearProjection(patches).to(self.device)
        b, n, _ = linear_projections.shape
        # cat with class token
        linear_projections = torch.cat((self.class_token, linear_projections), dim=1)
        # car pos embed
        pos_embed = einops.repeat(self.pos_embedding, "b 1 d -> b m d", m=n + 1)

        # add pos_embed to linear projection
        linear_projections += pos_embed
        # print(linear_projections.size())
        # print(pos_embed.size())

        return linear_projections
