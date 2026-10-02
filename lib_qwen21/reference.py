# Qwen-Image 2.1 support for Forge Neo
# reference images for editing: one resize feeds both the vision encoder and the VAE (diffusers QwenImage21Pipeline)

import hashlib
import math

import numpy as np
import torch
from PIL import Image

RESOLUTION = 1024
# one vision token covers 32x32 pixels, the 2x2 latent tokens of the VAE that take its place in the transformer
MULTIPLE = 32


def size(width: int, height: int, resolution: int = RESOLUTION) -> tuple[int, int]:
    # diffusers calculate_dimensions: the reference's aspect ratio at resolution², in multiples of 32
    ratio = width / height
    w = math.sqrt(resolution * resolution * ratio)
    h = w / ratio
    return max(MULTIPLE, round(w / MULTIPLE) * MULTIPLE), max(MULTIPLE, round(h / MULTIPLE) * MULTIPLE)


def resize(image: Image.Image, resolution: int = RESOLUTION) -> Image.Image:
    image = image.convert("RGBA")
    target = size(*image.size, resolution)
    if image.size == target:
        return image
    return image.resize(target, Image.Resampling.LANCZOS)


def vae_input(image: Image.Image) -> torch.Tensor:
    # (1, H, W, 4) in [0, 1]: the VAE reads the alpha channel too
    return torch.from_numpy(np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0).unsqueeze(0)


def vision_input(image: Image.Image) -> torch.Tensor:
    # (1, H, W, 3) in [0, 1]; the checkpoint was trained with the alpha composited over white for the vision encoder
    image = image.convert("RGBA")
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return torch.from_numpy(np.asarray(white, dtype=np.float32) / 255.0).unsqueeze(0)


def decorrelate(noise: torch.Tensor, images: list[Image.Image]) -> torch.Tensor:
    # with the seed that made the reference, the starting noise is the very noise the reference grew from: the
    # model reads the match as detail, re-renders a crunchy copy and barely edits. A spatial permutation keyed on
    # the reference keeps the noise Gaussian and the seed reproducible, and differs for chained edits too
    digest = hashlib.sha256(b"".join(np.asarray(image).tobytes() for image in images)).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)
    height, width = noise.shape[-2:]
    order = torch.randperm(height * width, generator=generator).to(noise.device)
    return noise.flatten(-2)[..., order].reshape(noise.shape)
