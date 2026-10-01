# Qwen-Image 2.1 support for Forge Neo
# transparent output: the alpha channel kept by the VAE is attached to each saved image

import numpy as np
from PIL import Image

from .engine import QwenImage21Engine

# an image whose alpha never drops below this stays RGB, exactly as before
OPAQUE = 250
# near-zero alpha is noise over undefined colour (faint specks in the background): cleared
FLOOR = 6
# the model also leaves small, nearly opaque islands of alpha in the background (4-190 px at 1 MP, against a subject
# of hundreds of thousands): an island below both limits is cleared, so a picture made only of small pieces keeps them
SPECK_AREA = 0.0003  # of the image
SPECK_RATIO = 0.01  # of the largest island


def is_qwen21(p) -> bool:
    return isinstance(getattr(p, "sd_model", None), QwenImage21Engine)


def requested(p) -> bool:
    # the decoded alpha of an opaque image still dips here and there (a few pixels at 200-250): only a request for
    # transparency, by the checkbox or by a prompt in the RGBA format, makes the saved image RGBA
    from .prompting import MARKERS

    if p.extra_generation_params.get("Transparent background", False):
        return True
    return any(m in str(getattr(p, "prompt", "")).lower() for m in MARKERS)


def remove_specks(a: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return a

    count, labels, stats, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8), connectivity=8)
    if count <= 2:
        return a
    areas = stats[1:, cv2.CC_STAT_AREA]
    limit = min(SPECK_AREA * a.size, SPECK_RATIO * areas.max())
    specks = np.flatnonzero(areas < limit) + 1
    if specks.size == 0:
        return a
    return np.where(np.isin(labels, specks), 0, a).astype(np.uint8)


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
    a = remove_specks(a)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(a, mode="L"))
    return rgba
