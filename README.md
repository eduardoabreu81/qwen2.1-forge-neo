# Qwen-Image 2.1 for Forge Neo

Run [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) in [Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) through the normal checkpoint, preset and Generate workflow.

The extension changes no Forge file. It registers the model at startup, in memory: delete the folder and Forge Neo is exactly as before.

## Requirements

- Forge Neo on the `neo` branch, updated on or after 2026-09-30 (the `TextProcessingEngine` rewrite).
- `comfy-kitchen` 0.2.36 or newer (Forge Neo installs it from its own requirements).

The extension checks the installed Forge Neo at startup. If a piece it relies on is missing, it changes nothing, leaves Qwen-Image 2.1 disabled and prints what is missing in the console.

## Models

| File | Folder | Source |
|---|---|---|
| a Qwen-Image 2.1 checkpoint, e.g. `qwen_image_2.1_int8_convrot.safetensors` or a merge such as [Qwen Image 2.1 Turbo 8 Step](https://civitai.com/models/2967502) | `models/Stable-diffusion` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models) |
| `qwen3vl_8b_int8_convrot.safetensors` (or `qwen3vl_8b_bf16` / `qwen3vl_8b_w4a8`) | `models/text_encoder` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/text_encoders) |
| `qwen_image_2.1_vae_bf16.safetensors` | `models/VAE` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) |

The ComfyUI `int8_convrot` / `w4a8` files load as they are, through Forge Neo's mixed-precision support.

## Usage

1. Pick the **qwen21** UI preset.
2. Select the checkpoint, then the text encoder and the VAE under **VAE / Text Encoder**.
3. Generate. The preset defaults to Euler, Simple, 8 steps and CFG 1: the model is distilled, and raising CFG burns the image.

Keep **Diffusion in Low Bits** on *Automatic*: pre-quantized checkpoints ignore it.

On GPUs without bf16 (RTX 20 series and older) the transformer and the VAE run in fp16, as in ComfyUI.

## Transparent images

Qwen-Image 2.1 decodes an alpha channel. Tick **Transparent background** in the **Qwen-Image 2.1** accordion and write the prompt as usual (`a cute cartoon dragon sticker`): the extension wraps it in the format Qwen recommends,

```
This is an RGBA image with transparency. A cute cartoon dragon sticker. The image has alpha channel and the background is transparent.
```

A prompt that already asks for an RGBA image is left as written, so the format can also be typed by hand.

Images with transparency are saved as RGBA PNG; fully opaque images stay RGB. JPG and WebP have no alpha, so save as PNG. The alpha is dropped when the image is resized after decoding (upscaler, face restoration) or when the VAE falls back to tiled decoding.

## Not supported yet

- LoRA
- image editing with reference images
- prefix K/V caching across steps

## Uninstalling

Switch the UI preset to another one first, then delete the extension folder. A saved `qwen21` preset has nothing to point to once the extension is gone.

## Credits and license

The model code is ported from [ComfyUI](https://github.com/Comfy-Org/ComfyUI) (GPL-3.0) and builds on [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic) (AGPL-3.0); this extension is released under the AGPL-3.0.

The Qwen-Image 2.1 weights are under the Qwen Research License Agreement and are not part of this repository.
