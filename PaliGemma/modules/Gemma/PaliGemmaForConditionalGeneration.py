import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math
import sys
import yaml

sys.path.append("/home/abdelrahman.elsayed/implementations/PaliGemma/modules/SigLip")
from SigLip import SiglipVisionModel


with open(
    "/home/abdelrahman.elsayed/implementations/PaliGemma/modules/SigLip/siglip_config.yaml",
    "r",
) as f:
    vision_config = yaml.safe_load(f)

print(f"Vision config : {vision_config}")


with open(
    "/home/abdelrahman.elsayed/implementations/PaliGemma/modules/Gemma/gemma_config.yaml",
    "r",
) as f:
    text_config = yaml.safe_load(f)

print(f"Text config : {text_config}")

with open(
    "/home/abdelrahman.elsayed/implementations/PaliGemma/modules/Gemma/paligemma_config.yaml",
    "r",
) as f:
    paligemma_config = yaml.safe_load(f)

print(f"PaliGemma config : {paligemma_config}")


text_config["GemmaConfig"]["vocab_size"] = paligemma_config["PaliGemmaConfig"][
    "vocab_size"
]
text_config["GemmaConfig"]["hidden_size"] = paligemma_config["PaliGemmaConfig"][
    "hidden_size"
]


class PaliGemmaMultiModalProjector(nn.Module):
    pass


class GemmaForCausalLM(nn.Module):
    pass


# The class that connects all of our components together
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config):
        super(PaliGemmaForConditionalGeneration, self).__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            self.config
        )  # Linear projection layer . Basically , we match the embeddings of the image to the text tokens dim so we can concatenate them
        self.vocab_size = self.config["PaliGemmaConfig"]["vocab_size"]

        language_model = GemmaForCausalLM(text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config["PaliGemmaConfig"]["pad_token_id"]
            if self.config["PaliGemmaConfig"]["pad_token_id"] is not None
            else -1
        )

    # Embedding Layer: Converts input words into vectors using an embedding matrix.
    # Softmax Layer: Converts the hidden states back into word probabilities using the same embedding matrix.
    # We tie weights , reducing the number of parameters of our model by sharing the weights of these two
    def tie_weights(self):
        return self.language_model.tie_weights()

    def merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: KVCache,
    ):
        _, _, embed_dim = (
            image_features.shape
        )  # note that the embed dim is the same as hidden_size for the text embeddings
        batch_size, sequence_length = input_ids.shape
        # example for input_ids [B-> number of sentence , sequence -> each id in the vocab for each word]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # shape: [B , Seq_len , Hidden_size]
        scaled_image_features = image_features / (
            self.config["PaliGemmaConfig"]["hidden_size"] ** 0.5
        )
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        # place holder to store the concatenated embedddings
        # NOTE: The sequence length is coming from the tokenizer
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=dtype, device=device
        )
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (
            input_ids != self.config["PaliGemmaConfig"]["image_token_index"]
        ) & (input_ids != self.config["PaliGemmaConfig"]["pad_token_id"])
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config["PaliGemmaConfig"]["image_token_index"]
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.config["PaliGemmaConfig"]["pad_token_id"]
        # These masks will tell us (0,1) the positions of text , pads , image tokens
        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(
            -1, -1, embed_dim
        )  # [B , seq_len , hidden_size]
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(
            -1, -1, embed_dim
        )  # [B , seq_len , hidden_size]
        image_mask_expanded = image_mask.unsqueeze(-1).expand(
            -1, -1, embed_dim
        )  # [B , num_patches , hidden_size]

        # Add the text embeddings
        final_embedding = torch.where(
            text_mask_expanded, inputs_embeds, final_embedding
        )
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        # actually it is less
        final_embedding = final_embedding.masked_scatter(
            image_mask_expanded, scaled_image_features
        )
        # Zero out padding tokens ( we don't have pad tokens in our imeplementations)
        final_embedding = torch.where(
            pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding
        )

        # create attention mask
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        # First we get the input embeddings from the language model
        # [B , seq_len , hidden_size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        # Second , we need the embedding from the vision model
        # [B , num_patches , embed_dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # We need to project these embeddings to have the same dimension as the text embeddings
        # [B , num_patches , hidden_size]
        image_features = self.multi_modal_projector(selected_image_feature)
        # Finally , we concat these to get the input for our lanugauage decoder
        inputs_embeds, attention_mask, position_ids = (
            self.merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids, attention_mask, kv_cache
            )
        )

        # Finally ,we pass these to the language model to get the outputs.
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
