# Qwen-Image 2.1 support for Forge Neo
# coexistence with the native support of Forge Neo (Haoming02/sd-webui-forge-classic#1512): when Forge Neo loads the
# model itself, the extension leaves the loading, the LoRA keys, the preset and the references to it, and adds only
# what Forge Neo does not have (schedules, degrid, transparent background, noise decorrelation, fp16 compute,
# a faster VAE)

REPO = "Qwen/Qwen-Image-2.1"
REFERENCE_OPTION = "qwen21_do_reference"
# where Forge Neo keeps its Qwen-Image 2.1 VAE (PR #1512)
VAE_MODULE = "backend.nn.qwen21_vae"

# set by the script at startup
ACTIVE = False

_engines: tuple[type, ...] | None = None


def _guess() -> type | None:
    from huggingface_guess import model_list

    return next((m for m in model_list.models if getattr(m, "huggingface_repo", None) == REPO), None)


def available() -> bool:
    # found by its model entry, not by module or class names, which may still change before the merge
    try:
        return _guess() is not None
    except Exception:
        return False


def engines() -> tuple[type, ...]:
    global _engines
    if _engines is None:
        from backend import loader

        guess = _guess()
        _engines = tuple(e for e in loader.possible_models if guess in getattr(e, "matched_guesses", []))
    return _engines


def _allow_fp16(guess: type) -> None:
    # Forge Neo declares bf16 / fp32 only, so GPUs without bf16 compute the transformer in fp32; its blocks clip to
    # the fp16 range as ours do, and fp16 is what the extension always ran there. The fp32 setting still asks for it
    from . import settings

    guess.supported_inference_dtypes = property(lambda self: settings.inference_dtypes())


def _fast_vae() -> None:
    # Forge Neo's VAE convolves through Conv3d with a temporal kernel of 1, which cuDNN runs far slower than the
    # equivalent Conv2d (minutes per decode on Turing), and in Forge Neo's VAE dtype, fp32 on GPUs without bf16.
    # The VAE is built when a model loads, so swapping the class it is built from and the dtype after the engine is
    # set up covers every load; nothing changes if the module is not there
    import importlib

    from . import vae

    module = importlib.import_module(VAE_MODULE)
    module.CausalConv3d = vae.CausalConv3d

    for engine in engines():
        original = engine.__init__
        if getattr(original, "qwen21_original", None):
            continue

        def __init__(self, *args, _original=original, **kwargs):
            _original(self, *args, **kwargs)
            patcher = self.forge_objects.vae
            dtype = vae.preferred_dtype()
            if patcher.vae_dtype != dtype:
                patcher.first_stage_model.to(dtype)
                patcher.vae_dtype = dtype

        __init__.qwen21_original = original
        engine.__init__ = __init__


def _tiled_rgba() -> None:
    # Forge Neo's tiled VAE decoding builds three-channel outputs whatever the VAE puts out, so the four channels of
    # Qwen-Image 2.1 crash it (Haoming02/sd-webui-forge-classic#1536): with "Enabled for VAE (always tiled)" and in the
    # fallback after running out of memory. A VAE of three channels and a Forge Neo that passes them itself keep the
    # original
    import inspect

    from backend.patcher import vae

    original = vae.VAE.decode_tiled_
    if getattr(original, "qwen21_original", None) or "out_channels" in inspect.getsource(original):
        return

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        channels = getattr(self, "output_channels", 3)
        if channels == 3:
            return original(self, samples, tile_x, tile_y, overlap)

        def decode_fn(a):
            return self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()

        tiles = ((tile_x // 2, tile_y * 2), (tile_x * 2, tile_y // 2), (tile_x, tile_y))
        output = sum(vae.tiled_scale(samples, decode_fn, x, y, overlap, upscale_amount=self.upscale_ratio, out_channels=channels, output_device=self.output_device) for x, y in tiles)
        return self.process_output(output / 3.0)

    decode_tiled_.qwen21_original = original
    vae.VAE.decode_tiled_ = decode_tiled_


def apply() -> None:
    from . import scheduler, settings

    try:
        settings.register()
        _allow_fp16(_guess())
    except Exception as e:
        print(f"[Qwen-Image 2.1] could not offer fp16 compute: {e}")

    try:
        _fast_vae()
    except Exception as e:
        print(f"[Qwen-Image 2.1] could not speed up the VAE: {e}")

    try:
        _tiled_rgba()
    except Exception as e:
        print(f"[Qwen-Image 2.1] could not fix the tiled VAE decoding: {e}")

    try:
        scheduler.register()
    except Exception as e:
        print(f"[Qwen-Image 2.1] could not add the {scheduler.LABEL} / {scheduler.BASE_LABEL} schedulers: {e}")
    print("[Qwen-Image 2.1] Forge Neo supports the model natively: the extension adds its schedules, degrid, transparent background, noise decorrelation, fp16 compute and a faster VAE")


def reference_on() -> bool:
    from modules import shared

    return bool(getattr(shared.opts, REFERENCE_OPTION, False))


def reference_images(p, gallery: list) -> list:
    # the images Forge Neo reads as references, as given: in img2img the input image comes first (<image1>) when its
    # reference setting is on, then the ImageStitch gallery; empty when the conditioning carries no reference
    from backend.args import dynamic_args

    if not getattr(dynamic_args, "ref_latents", None):
        return []
    images = []
    if getattr(p, "init_images", None) and reference_on():
        images.append(p.init_images[0])
    return images + list(gallery)
