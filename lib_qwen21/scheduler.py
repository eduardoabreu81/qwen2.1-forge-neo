# Qwen-Image 2.1 support for Forge Neo
# "Viggle Turbo" schedule: the timesteps the Viggle turbo LoRAs are distilled on, with the pipeline's
# resolution-dependent time shift. https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo
#
# 6 steps are the official nodes 1, 0.9375, 0.875, 0.75, 0.5, 0.25; other step counts add or remove nodes at the
# high-noise end only (1 .. 0.875), keeping 0.75, 0.5, 0.25, as Viggle's README asks

import math

import torch

NAME = "viggle_turbo"
LABEL = "Viggle Turbo"
# earlier name, so the infotext of images made with it still pastes back
ALIASES = ["Qwen 2.1 Turbo", "qwen21_turbo"]

LOW_NOISE = (0.75, 0.5, 0.25)
HIGH_NOISE = (1.0, 0.875)

# dynamic shift of the Qwen-Image 2.1 scheduler config
BASE_SEQ_LEN, MAX_SEQ_LEN = 256, 8192
BASE_SHIFT, MAX_SHIFT = 0.5, 0.9
PATCH = 16


def mu(width: int, height: int) -> float:
    tokens = (width // PATCH) * (height // PATCH)
    return BASE_SHIFT + (MAX_SHIFT - BASE_SHIFT) * (tokens - BASE_SEQ_LEN) / (MAX_SEQ_LEN - BASE_SEQ_LEN)


def timesteps(steps: int) -> list[float]:
    if steps < len(LOW_NOISE) + 1:
        # too few steps for the turbo nodes: uniform
        return torch.linspace(1.0, 0.0, steps + 1)[:-1].tolist()
    high = torch.linspace(HIGH_NOISE[0], HIGH_NOISE[1], steps - len(LOW_NOISE)).tolist()
    return high + list(LOW_NOISE)


def sigmas(steps: int, width: int, height: int) -> torch.Tensor:
    e = math.exp(mu(width, height))
    return torch.FloatTensor([e / (e + 1.0 / t - 1.0) for t in timesteps(steps)] + [0.0])


def _default(n, sigma_min=None, sigma_max=None, device="cpu", **kwargs):
    # used only when the script cannot pass the resolution: 1024x1024
    return sigmas(n, 1024, 1024).to(device)


def register() -> None:
    from modules import sd_schedulers, shared

    if LABEL in sd_schedulers.schedulers_map:
        return
    entry = sd_schedulers.Scheduler(NAME, LABEL, _default, aliases=list(ALIASES))
    sd_schedulers.all_schedulers.append(entry)
    if LABEL not in shared.opts.hide_schedulers:
        sd_schedulers.schedulers.append(entry)
        for key in (NAME, LABEL, *ALIASES):
            sd_schedulers.schedulers_map[key] = entry


def is_selected(p) -> bool:
    return getattr(p, "scheduler", None) in (NAME, LABEL, *ALIASES)


def override(p):
    # Forge passes the resolution only to its Flux2 scheduler; sampler_noise_scheduler_override is the public way in
    def schedule(steps: int) -> torch.Tensor:
        if getattr(p, "is_hr_pass", False):
            return sigmas(steps, p.hr_upscale_to_x, p.hr_upscale_to_y)
        return sigmas(steps, p.width, p.height)

    return schedule
