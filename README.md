# 🎨 Qwen-Image 2.1 for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Gradio](https://img.shields.io/badge/Gradio-4.40.0-orange)](https://gradio.app/)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/qwen2.1-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate with **[Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)** inside Forge Neo, using the checkpoint, preset and Generate button you already know - sharp text rendering, coherent scenes, transparent images and fast turbo checkpoints, even on 8 GB GPUs.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (the `neo` branch, updated on or after 30 September 2026). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [Installation](#-installation)
- [Recommended Settings](#%EF%B8%8F-recommended-settings)
- [Tips](#-tips)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.1.0 - First Release

- **Qwen-Image 2.1 in Forge Neo** - checkpoint, text encoder and VAE load through the usual **VAE / Text Encoder** selector, including the quantized files made for ComfyUI.
- **qwen21 UI preset** - sampler, schedule, steps and CFG ready for turbo checkpoints.
- **Viggle Turbo schedule** - the steps turbo checkpoints were trained on, for sharp images in 6-8 steps.
- **Transparent background** - one checkbox, and the image is saved as a PNG with real transparency.
- **Remove VAE grid** - cleans the faint dot-and-stripe pattern the Qwen-Image 2.1 VAE leaves on smooth areas.
- **LoRA support** - Qwen-Image 2.1 LoRAs with the usual `<lora:name:weight>` syntax.

---

## 🎯 Features

### 🖼️ Native Generation

- Runs inside txt2img - no separate tab, no extra environment
- Works with the Forge Neo model folders and the **VAE / Text Encoder** selector you already use
- Loads quantized checkpoints and text encoders as they are - tested on an 8 GB RTX 2070
- Negative prompt and CFG work as with any other model

### ⚡ Turbo Checkpoints

- **Viggle Turbo** schedule in the **Schedule type** list
- Adapts the steps to the image size automatically
- Works with any step count; 6-8 steps is the sweet spot

### 🎛️ Base Checkpoints

- **Qwen Base** schedule in the **Schedule type** list - the official schedule of the model, for base (non-turbo) checkpoints
- Adapts to the image size automatically

### 🧊 Transparent Images

- **Transparent background** checkbox in the **Qwen-Image 2.1** accordion
- Write only the subject - the extension adds the wording the model expects
- Saved as PNG with a real alpha channel; normal images stay exactly as before
- Stray specks the model leaves in the background are cleared; the subject and its soft shadow are kept

### 🧼 Clean Output

- **Remove VAE grid**, on by default, in the **Qwen-Image 2.1** accordion
- Touches only the repeating pattern; real texture and edges are left alone
- Images without the pattern pass through unchanged, including those from a fine-tuned VAE that no longer leaves it

### 🔌 LoRA

- Qwen-Image 2.1 LoRAs through Forge's usual `<lora:name:weight>` syntax

### 🛡️ Safe by Design

- Changes no Forge Neo file - remove the extension and everything is as it was
- Checks your Forge Neo at startup and stays disabled if something it needs is missing
- Other models are not affected

---

## 📦 Installation

1. Open Forge Neo WebUI
2. Go to **Extensions** → **Install from URL**
3. Paste: `https://github.com/eduardoabreu81/qwen2.1-forge-neo`
4. Click **Install** and restart the WebUI
5. Place your Qwen-Image 2.1 files in the usual folders:

| Part | Folder |
| :--- | :--- |
| Qwen-Image 2.1 checkpoint | `models/Stable-diffusion` |
| Qwen3-VL 8B text encoder | `models/text_encoder` |
| Qwen-Image 2.1 VAE | `models/VAE` |

6. Pick the **qwen21** UI preset, the checkpoint, and **both** the text encoder and the VAE under **VAE / Text Encoder**

---

## ⚙️ Recommended Settings

| Checkpoint | Sampler | Schedule type | Steps | CFG |
| :--- | :--- | :--- | :--- | :--- |
| **Turbo** | Euler | Viggle Turbo | 6-8 | 1 |
| **Base** (non-turbo) | Euler or Res Multistep | Qwen Base, Simple or Beta | 16-25 | 2-3, with a negative prompt |
| **Base** + turbo LoRA at weight 1 | Euler | Viggle Turbo | 6 | 1 |

---

## 💡 Tips

- Keep **Diffusion in Low Bits** on *Automatic*
- To keep **Viggle Turbo** after switching presets, set it in **Settings** → **Presets** → **QWEN21**
- **Viggle Turbo** is for turbo checkpoints only - on a base checkpoint it leaves grain in fine detail such as hair, at any step count; use **Qwen Base** or **Simple** there
- With CFG above 1, **Settings** → **Optimizations** → **Skip Negative Prompt during Later Steps** at 0.5, with **For the above option, skip every step** on, makes a base checkpoint about 12% faster with a near-identical image
- Many style LoRAs are made for the base model - on a turbo checkpoint, try a higher weight (1.2-1.5) or use them with a base checkpoint
- Use the trigger word a LoRA asks for
- Transparent images need **PNG**; upscalers and face restoration remove the transparency
- Before uninstalling, switch the UI preset to another one

**Not supported yet:** image editing with reference images.

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- **[Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)** by the Qwen team
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - reference implementation of the model
- **[Viggle](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)** - turbo schedule
- **[ComfyUI-DeGrid](https://github.com/lunaaispace-eng/ComfyUI-DeGrid)** by lunaaispace-eng - grid detection idea

---

## 📜 License

AGPL-3.0 - see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/qwen2.1-forge-neo/issues)** • **[Request Feature](https://github.com/eduardoabreu81/qwen2.1-forge-neo/issues)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
