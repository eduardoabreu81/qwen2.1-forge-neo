# Qwen-Image 2.1 support for Forge Neo
# runtime hooks: every hook hands anything that is not Qwen-Image 2.1 to the original Forge Neo function untouched

import contextlib
import logging
import os

import torch
from transformers.modeling_utils import no_init_weights

from backend import loader, memory_management
from backend.nn import krea
from backend.nn.llm import llama
from backend.operations import using_forge_operations
from backend.state_dict import load_state_dict
from huggingface_guess import detection, model_list

from . import lora, model, presets
from .engine import QwenImage21Engine
from .text_encoder import Qwen3VL8B
from .vae import QwenImage21VAE

logger = logging.getLogger("qwen21")

TE_DEEPSTACK_KEY = "model.visual.deepstack_merger_list.0.norm.weight"
TE_MERGER_KEY = "model.visual.merger.linear_fc2.weight"
TE_8B_DIM = 4096

_applied = False


@contextlib.contextmanager
def _swapped(module, name: str, value):
    # the loader imports these classes inside the function, so a temporary swap routes one call to our class
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def _is_qwen3vl_8b(asd: dict) -> bool:
    return TE_DEEPSTACK_KEY in asd and TE_MERGER_KEY in asd and int(asd[TE_MERGER_KEY].shape[0]) == TE_8B_DIM


MISSING_VAE = "Qwen-Image 2.1 needs its own VAE: select qwen_image_2.1_vae_bf16.safetensors under VAE / Text Encoder"
MISSING_TE = "Qwen-Image 2.1 needs the Qwen3-VL 8B text encoder: select qwen3vl_8b_*.safetensors under VAE / Text Encoder"


def _load_vae(config_path: str, state_dict: dict) -> QwenImage21VAE:
    if not isinstance(state_dict, dict) or "decoder.head.2.weight" not in state_dict:
        raise ValueError(MISSING_VAE)

    config = QwenImage21VAE.load_config(config_path)
    config["in_channels"] = int(state_dict["encoder.conv1.weight"].shape[1])
    config["out_channels"] = int(state_dict["decoder.head.2.weight"].shape[0])
    config["base_dim"] = int(state_dict["encoder.conv1.weight"].shape[0])
    config["decoder_base_dim"] = int(state_dict["decoder.head.0.gamma"].shape[0])

    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype(), extra_dtype="vae"):
            vae = QwenImage21VAE.from_config(config)

    load_state_dict(vae, state_dict)
    return vae


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_replace_state_dict() -> None:
    original = loader.replace_state_dict

    def replace_state_dict(sd: dict, asd: dict, guess, path):
        # Forge Neo only accepts the 4B Qwen3-VL (Krea 2); the 8B one is filed under its own key
        if not _is_qwen3vl_8b(asd):
            return original(sd, asd, guess, path)
        prefix = f"{guess.text_encoder_key_prefix[0]}qwen3vl_8b."
        for k in [k for k in sd if k.startswith(prefix)]:
            del sd[k]
        for k, v in asd.items():
            sd[f"{prefix}transformer.{k}"] = v
        return sd

    loader.replace_state_dict = replace_state_dict


def _hook_components() -> None:
    original = loader.load_huggingface_component

    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.QwenImage21):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "AutoencoderKLQwenImage21":
            return _load_vae(os.path.join(repo_path, component_name), state_dict)
        if cls_name == "QwenImage21Transformer2DModel":
            # the Krea 2 branch is a generic single-stream DiT load: dtype, quantization and device handling
            from .transformer import QwenImage21Transformer2DModel

            with _swapped(krea, "SingleStreamDiT", QwenImage21Transformer2DModel):
                return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
        if cls_name == "Qwen3VLModel":
            if not isinstance(state_dict, dict) or len(state_dict) <= 16:
                raise ValueError(MISSING_TE)
            with _swapped(llama, "Qwen3VL", Qwen3VL8B):
                return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.QwenImage21 not in model_list.models:
        model_list.models.append(model.QwenImage21)
    if QwenImage21Engine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, QwenImage21Engine)
    _hook_replace_state_dict()
    _hook_components()
    lora.hook()

    try:
        presets.register()
    except Exception as e:
        # the model still loads without the preset; any other preset works with Euler / Simple / 8 steps / CFG 1
        logger.warning(f"[Qwen-Image 2.1] could not add the qwen21 UI preset: {e}")

    _applied = True
    print(f"[Qwen-Image 2.1] enabled (torch {torch.__version__})")
