# Qwen-Image 2.1 support for Forge Neo
# extension settings, under Settings > Stable Diffusion > Qwen-Image 2.1

import torch

FP32_COMPUTE = "qwen21_fp32_compute"

SECTION = ("qwen21", "Qwen-Image 2.1", "sd")


def register() -> None:
    from modules import shared
    from modules.options import OptionInfo, options_section

    options = options_section(
        SECTION,
        {
            FP32_COMPUTE: OptionInfo(False, "Compute the transformer in fp32 on GPUs without bf16").info("exact, as ComfyUI does on those GPUs; slower and heavier than fp16. Takes effect on the next generation"),
        },
    )
    for key, info in options.items():
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)
    shared.opts.onchange(FP32_COMPUTE, _reload_model, call=False)


def _reload_model() -> None:
    # the compute dtype is fixed at load time: an empty hash makes Forge reload the model on the next generation
    from modules import sd_models

    sd_models.model_data.forge_hash = ""


def inference_dtypes() -> list[torch.dtype]:
    # ComfyUI declares bf16 / fp32 only; fp16 stays available (the blocks clip to its range) unless fp32 is asked for
    from modules import shared

    if getattr(shared.opts, FP32_COMPUTE, False):
        return [torch.bfloat16, torch.float32]
    return [torch.bfloat16, torch.float16, torch.float32]
