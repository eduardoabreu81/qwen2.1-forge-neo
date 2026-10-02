# Qwen-Image 2.1 support for Forge Neo
# diffusion engine: text-to-image and img2img

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.args import args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE

from . import reference
from .model import QwenImage21
from .text_engine import Qwen3VL8BEngine

VAE_SCALE = 16


def _vae_dtype() -> torch.dtype:
    # ComfyUI runs this VAE in bf16 / fp16 / fp32 (in that order); the global Forge default skips fp16 because of
    # the SD VAEs, which leaves GPUs without bf16 decoding in fp32 at twice the memory
    if args.fp16_vae or args.bf16_vae or args.fp32_vae:
        return memory_management.vae_dtype()
    device = memory_management.vae_device()
    if memory_management.should_use_bf16(device):
        return torch.bfloat16
    if memory_management.should_use_fp16(device):
        return torch.float16
    return torch.float32


def _qwen21_vae(model: torch.nn.Module) -> VAE:
    vae = VAE(model=model, dtype=_vae_dtype())
    # 2d, 16x, 64 channels; memory estimates from ComfyUI
    vae.upscale_ratio = VAE_SCALE
    vae.downscale_ratio = VAE_SCALE
    vae.latent_channels = int(model.config.z_dim)
    vae.memory_used_encode = lambda shape, dtype: (600 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
    vae.memory_used_decode = lambda shape, dtype: (900 * shape[2] * shape[3] * (VAE_SCALE * VAE_SCALE)) * memory_management.dtype_size(dtype)
    return vae


class QwenImage21Engine(ForgeDiffusionEngine):
    matched_guesses = [QwenImage21]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_8b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_8b": huggingface_components["tokenizer"]})

        vae = _qwen21_vae(huggingface_components["vae"])

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen3VL8BEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_8b,
            tokenizer=clip.tokenizer.qwen3vl_8b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        # alpha of the last decoded batch, (B, 1, H, W) in [-1, 1]; applied to the saved images by the script
        self.last_alpha: torch.Tensor | None = None

        # reference images of the current generation, resized; the vision encoder reads them with both the prompt
        # and the negative prompt, as the diffusers pipeline does
        self.references: list["Image.Image"] = []
        # the images as given and their resolution, to set them again with one of them changed
        self.reference_sources: tuple[list["Image.Image"], int] = ([], reference.RESOLUTION)

    @torch.inference_mode()
    def set_references(self, images: list["Image.Image"], resolution: int = reference.RESOLUTION) -> None:
        # called by the script for every generation; one resize feeds both the vision encoder and the VAE
        self.references = [reference.resize(image, resolution) for image in images]
        self.reference_sources = (list(images), resolution)
        diffusion_model = self.forge_objects.unet.model.diffusion_model
        diffusion_model.reference_latents = [self._encode_reference(image) for image in self.references]
        diffusion_model.image_slots = []

    def _encode_reference(self, image: "Image.Image") -> torch.Tensor:
        # normalized like the target latents (diffusers _encode_vae_image)
        vae = self.forge_objects.vae
        return vae.first_stage_model.process_in(vae.encode(reference.vae_input(image))).cpu()

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor):
        vae_model = self.forge_objects.vae.first_stage_model
        vae_model.alpha_target_hw = (x.shape[-2] * VAE_SCALE, x.shape[-1] * VAE_SCALE)
        vae_model.alpha_chunks = []
        try:
            sample = super().decode_first_stage(x)
        finally:
            vae_model.alpha_target_hw = None

        chunks, vae_model.alpha_chunks = vae_model.alpha_chunks, []
        alpha = torch.cat(chunks) if chunks else None
        # a tiled fallback keeps no alpha: the batch then stays RGB
        self.last_alpha = alpha if alpha is not None and alpha.shape[0] == x.shape[0] else None
        return sample

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        images = [reference.vision_input(image) for image in self.references]
        cond = self.text_processing_engine_qwen(prompt, images=images)
        # the same for the prompt and the negative prompt: the images come before either text
        self.forge_objects.unet.model.diffusion_model.image_slots = self.text_processing_engine_qwen.image_slots
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)
