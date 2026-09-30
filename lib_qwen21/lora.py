# Qwen-Image 2.1 support for Forge Neo
# LoRA keys: Forge Neo's "qwen" mapping already covers the Qwen-Image 2.1 layer names; what it cannot know is that
# a comfy-saved checkpoint fuses the MLP gate and up projections into img_mlp.gate_up, while LoRAs address the
# halves as img_mlp.gate_layer and img_mlp.proj (the same split ComfyUI makes in comfy/lora.py)

import sys

from .model import QwenImage21

FUNCTION = "model_lora_keys_unet"


def _halves(key_lora: str) -> list[str]:
    flat = key_lora.replace(".", "_")
    return [key_lora, f"transformer.{key_lora}", f"diffusion_model.{key_lora}", f"lycoris_{flat}", f"lora_unet_{flat}", f"lora_transformer_{flat}"]


def add_fused_mlp_keys(model, key_map: dict) -> dict:
    config: dict = getattr(model.diffusion_model, "config", {}) or {}
    if not config.get("fused_mlp", False):
        return key_map

    half = int(config["num_attention_heads"]) * int(config["attention_head_dim"]) * int(config["mlp_ratio"])
    for i in range(int(config["num_layers"])):
        fused = f"diffusion_model.transformer_blocks.{i}.img_mlp.gate_up.weight"
        # [gate; up] rows: gate_layer is the first half, proj the second
        for name, offset in (("gate_layer", (0, 0, half)), ("proj", (0, half, half))):
            for key in _halves(f"transformer_blocks.{i}.img_mlp.{name}"):
                key_map[key] = (fused, offset)
    return key_map


def wrap(original):
    def model_lora_keys_unet(model, key_map=None):
        # called exactly as before, including the original's own default, so other models see no change
        key_map = original(model) if key_map is None else original(model, key_map)
        if isinstance(getattr(model, "config", None), QwenImage21):
            key_map = add_fused_mlp_keys(model, key_map)
        return key_map

    model_lora_keys_unet.qwen21_original = original
    return model_lora_keys_unet


def hook() -> int:
    # the LoRA extension imports the function by name, so every module holding the original gets the wrapper;
    # safe to call again: already wrapped references are skipped
    from modules_forge.packages.comfy import lora as comfy_lora

    original = getattr(comfy_lora, FUNCTION)
    if hasattr(original, "qwen21_original"):
        original = original.qwen21_original
    wrapped = None
    count = 0
    for module in list(sys.modules.values()):
        try:
            found = getattr(module, FUNCTION, None) is original
        except Exception:
            continue
        if found:
            if wrapped is None:
                wrapped = wrap(original)
            setattr(module, FUNCTION, wrapped)
            count += 1
    return count
