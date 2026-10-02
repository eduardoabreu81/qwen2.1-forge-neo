# Qwen-Image 2.1 support for Forge Neo
# checks that the installed Forge Neo has every piece this extension builds on, before anything is patched

import importlib
import importlib.metadata
import inspect

# what the extension imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.text_processing._comfy": ("SDClipModel", "SDTokenizer", "INF", "EMBEDDINGS", "TOKEN_WEIGHTS"),
    "backend.text_processing.emphasis": ("EmphasisNone", "uses_emphasis"),
    "backend.nn.llm.llama": ("Qwen3VL", "Qwen3VL_4BConfig", "Llama2_", "attention_function"),
    "backend.nn.llm.qwen_vl": ("process_qwen2vl_images", "qwen2vl_mrope_position_ids"),
    "backend.nn.llm.qwen35": ("QWEN3VL_VISION", "Qwen3VLVisionModel"),
    "backend.nn.krea": ("SingleStreamDiT",),
    "backend.nn.flux": ("EmbedND", "timestep_embedding"),
    "backend.nn.wan_vae": ("CausalConv3d", "AttentionBlock", "RMS_norm"),
    "backend.nn._vae": ("ProcessLatent",),
    "backend.operations": ("get_weight_and_bias", "main_stream_worker", "weights_manual_cast", "using_forge_operations"),
    "backend.quant_ops": ("ck", "QUANT_ALGOS"),
    "backend.state_dict": ("load_state_dict",),
    "backend.loader": ("load_huggingface_component", "replace_state_dict", "possible_models"),
    "backend.patcher.vae": ("VAE",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine", "ForgeObjects"),
    "huggingface_guess.detection": ("detect_unet_config", "count_blocks"),
    "huggingface_guess.model_list": ("models", "BASE", "ModelType"),
    "huggingface_guess.latent": ("LatentFormat",),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "register"),
    "modules_forge.packages.comfy.lora": ("model_lora_keys_unet",),
    "modules.options": ("OptionInfo", "options_section"),
    "modules.processing": ("StableDiffusionProcessing",),
}

# methods the extension calls on Forge Neo classes, by module and class
REQUIRED_METHODS = {
    ("backend.nn.llm.llama", "Qwen3VL"): ("preprocess_embed", "build_image_inputs"),
    ("modules.processing", "StableDiffusionProcessing"): ("clear_prompt_cache",),
}

# the transformer and the text encoder go through these branches of the loader
LOADER_BRANCHES = ("Krea2Transformer2DModel", "Qwen3VLModel")

MIN_COMFY_KITCHEN = (0, 2, 36)


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for p in version.split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        for (module_name, class_name), names in REQUIRED_METHODS.items():
            cls = getattr(importlib.import_module(module_name), class_name)
            missing = [n for n in names if not hasattr(cls, n)]
            if missing:
                problems.append(f"{module_name}.{class_name} is missing {', '.join(missing)}")

        from backend import loader
        from backend.quant_ops import QUANT_ALGOS, ck

        source = inspect.getsource(loader.load_huggingface_component)
        missing = [b for b in LOADER_BRANCHES if f'"{b}"' not in source]
        if missing:
            problems.append(f"backend.loader has no {', '.join(missing)} branch")
        if "int8_tensorwise" not in QUANT_ALGOS:
            problems.append("backend.quant_ops does not support int8_tensorwise")
        if not hasattr(ck, "apply_rope"):
            problems.append("comfy_kitchen has no apply_rope")

    try:
        version = importlib.metadata.version("comfy-kitchen")
        if _version_tuple(version) < MIN_COMFY_KITCHEN:
            problems.append(f"comfy-kitchen {version} is older than {'.'.join(map(str, MIN_COMFY_KITCHEN))}")
    except importlib.metadata.PackageNotFoundError:
        problems.append("comfy-kitchen is not installed")

    return problems


def report(problems: list[str]) -> str:
    lines = [
        "[Qwen-Image 2.1] This Forge Neo is too old for the extension, so nothing was changed and Qwen-Image 2.1 is disabled.",
        "[Qwen-Image 2.1] Update Forge Neo (git pull on the neo branch) and restart. Details:",
    ]
    lines += [f"[Qwen-Image 2.1]   - {p}" for p in problems]
    message = "\n".join(lines)
    print(message)
    return message
