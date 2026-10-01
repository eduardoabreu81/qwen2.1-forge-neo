# Qwen-Image 2.1 support for Forge Neo
# transparent output: the alpha channel kept by the VAE is attached to each saved image

import numpy as np
from PIL import Image

from .engine import QwenImage21Engine

# an image whose alpha never drops below this stays RGB, exactly as before
OPAQUE = 250
# near-zero alpha is noise over undefined colour (faint specks in the background): cleared
FLOOR = 6


def is_qwen21(p) -> bool:
    return isinstance(getattr(p, "sd_model", None), QwenImage21Engine)


def requested(p) -> bool:
    # the decoded alpha of an opaque image still dips here and there (a few pixels at 200-250): only a request for
    # transparency, by the checkbox or by a prompt in the RGBA format, makes the saved image RGBA
    from .prompting import MARKERS

    if p.extra_generation_params.get("Transparent background", False):
        return True
    return any(m in str(getattr(p, "prompt", "")).lower() for m in MARKERS)


def apply(p, image: Image.Image) -> Image.Image:
    if not is_qwen21(p) or not requested(p):
        return image
    engine = p.sd_model

    alpha = engine.last_alpha
    index = getattr(p, "batch_index", 0)
    if alpha is None or index >= alpha.shape[0]:
        return image

    a = alpha[index, 0].clamp(-1.0, 1.0).add(1.0).mul(127.5).round().numpy().astype(np.uint8)
    if int(a.min()) >= OPAQUE:
        return image
    if (a.shape[1], a.shape[0]) != image.size:
        # resized after decoding (upscaler, face restoration...): the mask no longer lines up
        return image

    a[a < FLOOR] = 0
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(a, mode="L"))
    return rgba
