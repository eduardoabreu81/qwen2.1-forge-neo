# Qwen-Image 2.1 support for Forge Neo
# removes the 2-pixel dot grid the Qwen-Image 2.1 VAE leaves in its output
#
# The grid's phase is tied to the VAE's output stride, so within a neighbourhood it repeats at the same (y%2, x%2)
# position, while real detail has no fixed phase. Per tile, the median of the fine detail on each of the four
# sublattices estimates the grid; real texture cancels out. The estimate is interpolated between tile centres
# and subtracted, so nothing but the phase-locked component is touched. Same detection idea as ComfyUI-DeGrid
# (https://github.com/lunaaispace-eng/ComfyUI-DeGrid), a narrower correction.

import numpy as np
from PIL import Image

TILE = 64
# the VAE leaves a 2-pixel dot grid and 4-pixel stripes; a 4x4 phase pattern covers both
PERIOD = 4
# grid amplitude (grey levels) below which an image is left untouched; DeGrid uses 0.5/255
THRESHOLD = 0.5
# the correction is a few grey levels at most: anything larger is not the VAE grid
LIMIT = 4.0

# [1 2 2 2 1] / 8: zero response at the 2- and 4-pixel frequencies, so both patterns stay whole in x - blur
_KERNEL = np.array([1, 2, 2, 2, 1], dtype=np.float32) / 8.0


def _binomial_blur(x: np.ndarray) -> np.ndarray:
    r = len(_KERNEL) // 2
    p = np.pad(x, ((r, r), (r, r), (0, 0)), mode="reflect")
    h = sum(w * p[:, i : i + x.shape[1]] for i, w in enumerate(_KERNEL))
    return sum(w * h[i : i + x.shape[0]] for i, w in enumerate(_KERNEL))


def _tile_patterns(detail: np.ndarray) -> np.ndarray:
    # (tiles_y, tiles_x, P, P, channels): the zero-mean PxP pattern of each tile
    H, W, C = detail.shape
    ty, tx = H // TILE, W // TILE
    d = detail[: ty * TILE, : tx * TILE].reshape(ty, TILE // PERIOD, PERIOD, tx, TILE // PERIOD, PERIOD, C)
    pattern = np.median(d, axis=(1, 4))  # (ty, P, tx, P, C)
    pattern = pattern.transpose(0, 2, 1, 3, 4)
    return pattern - pattern.mean(axis=(2, 3), keepdims=True)


def _interpolate(values: np.ndarray, pos: np.ndarray) -> np.ndarray:
    # linear interpolation of per-tile values (axis 0) at pixel positions, anchored at tile centres
    if values.shape[0] == 1:
        return np.repeat(values, len(pos), axis=0)
    centres = (np.arange(values.shape[0]) + 0.5) * TILE
    i = np.clip(np.searchsorted(centres, pos + 0.5) - 1, 0, values.shape[0] - 2)
    w = np.clip((pos + 0.5 - centres[i]) / TILE, 0.0, 1.0).reshape(-1, *([1] * (values.ndim - 1)))
    return values[i] * (1 - w) + values[i + 1] * w


def amplitude(image: Image.Image) -> float:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    detail = rgb - _binomial_blur(rgb)
    H, W = (rgb.shape[0] // PERIOD) * PERIOD, (rgb.shape[1] // PERIOD) * PERIOD
    phases = detail[:H, :W].reshape(H // PERIOD, PERIOD, W // PERIOD, PERIOD, 3).mean(axis=(0, 2))
    return float(np.abs(phases - phases.mean(axis=(0, 1))).max())


def remove(image: Image.Image) -> Image.Image:
    if image.width < TILE or image.height < TILE or amplitude(image) < THRESHOLD:
        return image

    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    H, W, _ = rgb.shape

    pattern = _tile_patterns(rgb - _binomial_blur(rgb))  # (ty, tx, P, P, C)

    # one sublattice at a time: interpolate that phase's per-tile value only where it applies
    out = rgb.copy()
    for py in range(PERIOD):
        ys = np.arange(py, H, PERIOD)
        for px in range(PERIOD):
            xs = np.arange(px, W, PERIOD)
            rows = _interpolate(pattern[:, :, py, px], ys)  # (len(ys), tx, C)
            correction = _interpolate(rows.transpose(1, 0, 2), xs).transpose(1, 0, 2)  # (len(ys), len(xs), C)
            out[py::PERIOD, px::PERIOD] -= np.clip(correction, -LIMIT, LIMIT)

    out = np.clip(out, 0, 255).round().astype(np.uint8)
    result = Image.fromarray(out, mode="RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result
