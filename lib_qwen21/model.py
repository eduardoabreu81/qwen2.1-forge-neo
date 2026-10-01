# Qwen-Image 2.1 support for Forge Neo
# model registration for huggingface_guess, ported from ComfyUI (latent_formats / model_detection / supported_models)

import os

import torch
from huggingface_guess.detection import count_blocks
from huggingface_guess.latent import LatentFormat
from huggingface_guess.model_list import BASE, ModelType

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "Qwen-Image-2.1")

REQUIRED_KEYS = ("txt_in.text_norm.weight", "modulation.1.weight", "transformer_blocks.0.attn.norm_q.weight", "img_in.weight", "proj_out.weight")

class QwenImage21Latent(LatentFormat):
    def __init__(self):
        self.latent_channels = 64
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.0158, -0.0115, -0.0174], [ 0.0030,  0.0120,  0.0027], [ 0.0637,  0.0470, -0.0127], [ 0.0360,  0.0661, -0.0030],
            [ 0.0159,  0.0181,  0.0082], [ 0.0132,  0.0326,  0.0169], [ 0.0191,  0.0261,  0.0136], [-0.0146, -0.0276, -0.0361],
            [ 0.0187, -0.0024, -0.0072], [-0.1059, -0.0090,  0.0350], [-0.0195, -0.0226, -0.0138], [-0.0295,  0.0024, -0.0215],
            [ 0.0191, -0.0393, -0.0001], [-0.0144, -0.0166, -0.0272], [ 0.0389,  0.0430,  0.0445], [-0.0153, -0.0336,  0.0031],
            [ 0.0339,  0.0122,  0.0220], [-0.0136, -0.0078, -0.0120], [-0.0340, -0.0282, -0.0245], [-0.0133, -0.0176, -0.0133],
            [ 0.0109, -0.0087,  0.0096], [-0.0010,  0.0044,  0.0016], [ 0.0301,  0.0053,  0.0361], [-0.0281, -0.0205, -0.0032],
            [-0.0725,  0.0002,  0.0160], [-0.0036,  0.0158,  0.0807], [ 0.0087,  0.0040, -0.0053], [-0.0260,  0.0183, -0.0077],
            [-0.0039, -0.0035, -0.0107], [-0.0026,  0.0172,  0.0237], [ 0.0088,  0.0078,  0.0078], [-0.0087, -0.0310, -0.0122],
            [-0.0027,  0.0018,  0.0094], [-0.0064,  0.0292, -0.0256], [ 0.0594,  0.1049,  0.1180], [ 0.0103, -0.0103, -0.0026],
            [-0.0091,  0.0025, -0.0015], [ 0.0178,  0.0243,  0.0292], [-0.0063, -0.0012,  0.0202], [ 0.0452,  0.0246,  0.0143],
            [ 0.0149,  0.0270,  0.0052], [ 0.1484,  0.0801,  0.0804], [-0.0120,  0.0040,  0.0010], [ 0.0181,  0.0051, -0.0021],
            [ 0.0132,  0.0050,  0.0019], [ 0.0291,  0.0020,  0.0092], [ 0.0066, -0.0410, -0.1314], [-0.1153, -0.0629, -0.0802],
            [ 0.0258,  0.0378,  0.0298], [ 0.0375,  0.1139,  0.0468], [-0.0142, -0.0126, -0.0276], [ 0.0339,  0.0153,  0.0138],
            [ 0.0346,  0.0211,  0.0267], [ 0.0369, -0.0431, -0.0993], [-0.0052, -0.0092,  0.0056], [-0.0279,  0.0410, -0.0357],
            [ 0.0036,  0.0017, -0.0083], [-0.0441, -0.0367, -0.0454], [-0.0001, -0.0092, -0.0001], [-0.0222, -0.0183, -0.0051],
            [ 0.0039,  0.0053, -0.0184], [-0.0094, -0.0075, -0.0143], [-0.0066, -0.0088, -0.0063], [ 0.0220,  0.0074,  0.0100],
        ]
        self.latent_rgb_factors_bias = [-0.1228, -0.1869, -0.3083]

        self.latents_mean = torch.tensor([
            0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
            1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
            2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
            -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
            -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
            -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
            -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
            0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
        ]).view(1, self.latent_channels, 1, 1)
        self.latents_std = torch.tensor([
            3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
            3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
            3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
            3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
            3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
            3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
            3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
            3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
        ]).view(1, self.latent_channels, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std + latents_mean


class QwenImage21(BASE):
    # absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part
    huggingface_repo = CONFIG_DIR

    unet_config = {
        "image_model": "qwen_image21",
    }

    # scheduler mu at 1024x1024 (base 0.5 @ 256 tokens, max 0.9 @ 8192)
    sampling_settings = {
        "multiplier": 1.0,
        "shift": 0.69,
    }

    memory_usage_factor = 6.0

    unet_extra_config = {}
    latent_format = QwenImage21Latent

    @property
    def supported_inference_dtypes(self) -> list[torch.dtype]:
        # read when the model loads: fp16 unless Settings ask for fp32 on GPUs without bf16
        from .settings import inference_dtypes

        return inference_dtypes()

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {"qwen3vl_8b.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLUX


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    state_dict_keys = list(state_dict.keys())
    if not all(f"{key_prefix}{k}" in state_dict for k in REQUIRED_KEYS):
        return None
    if not any(f"{key_prefix}transformer_blocks.0.img_mlp.{k}.weight" in state_dict for k in ("gate_up", "proj")):
        return None

    dit_config = {"image_model": "qwen_image21"}
    head_dim = int(state_dict[f"{key_prefix}transformer_blocks.0.attn.norm_q.weight"].shape[0])
    inner_dim = int(state_dict[f"{key_prefix}img_in.weight"].shape[0])
    dit_config["in_channels"] = int(state_dict[f"{key_prefix}img_in.weight"].shape[1])
    dit_config["out_channels"] = int(state_dict[f"{key_prefix}proj_out.weight"].shape[0])
    dit_config["num_layers"] = count_blocks(state_dict_keys, f"{key_prefix}transformer_blocks." + "{}.")
    dit_config["attention_head_dim"] = head_dim
    dit_config["num_attention_heads"] = inner_dim // head_dim
    dit_config["context_in_dim"] = int(state_dict[f"{key_prefix}txt_in.text_norm.weight"].shape[0])
    # a comfy-saved file has gate and up fused into one GEMM; the diffusers layout keeps them apart
    gate_up = state_dict.get(f"{key_prefix}transformer_blocks.0.img_mlp.gate_up.weight", None)
    if gate_up is not None:
        dit_config["mlp_ratio"] = int(gate_up.shape[0]) // 2 // inner_dim
    else:
        dit_config["mlp_ratio"] = int(state_dict[f"{key_prefix}transformer_blocks.0.img_mlp.proj.weight"].shape[0]) // inner_dim
    dit_config["fused_mlp"] = gate_up is not None
    return dit_config