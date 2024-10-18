from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


# Implement the preprocessing functions used to process the images
def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.arrau(std, dtype=image.dtyp)
    image = (image - mean) / std
    return image


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    # convert to numpy array
    images = [np.array(image) for image in images]
    # Rescale the pixel values to be between [0,1]
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the images mean = 0 , std = 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # move the channel dimension to the dim 0
    images = [image.transpose(2, 0, 1) for image in images]

    return images

# This is taken from Hugging Face implementaion (Although in the paper , it was suggested that the sep token is tokenized separately)
# Here they are combining it with the text prompt
#ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"  # placeholder for image tokens

    def __init__(self, tokenizer, num_images_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_images_tokens
        self.image_size = image_size

        # Tokenizer described here: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md#tokenizer
        # PaliGemma supports segmentation and bounding_boxes generation
        # Thats why we have these special tokens!
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # These tokens are used for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # These tokens are used for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        # BOS and EOS tokens will be added later
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
    def __call__(self , text:List[str] , images: List[Image.Image] , padding: str="longest" , truncation:bool = True)->dict:
        # WE are now accepting for inference only one image and one text
        assert len(images) == 1 and len(text) == 1 , f"Recieved {len(images)} images and {len(text)} prompts"
        inputs = process_images(
            images , 
            size=(self.image_size , self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor= 1/255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD
        )

        # convert the list of arrays into single array -> [B , C , H , W]
        inputs = np.stack(inputs , axis=0)
        # convert to tensor to be ready for the vision encoder
        inputs = torch.tensor(inputs)
        # accoring to the paper and implmentation
        # we aprepend a seq_length number of image tokens(place_holders) to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token= self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token= self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Returns the input ids ( in the vocab) and attention mask
        inputs_text = self.tokenizer(input_strings,return_tensors="pt" , padding=padding , truncation = truncation)

        return_data = {"pixel_values" : inputs , **inputs_text}

        return return_data
