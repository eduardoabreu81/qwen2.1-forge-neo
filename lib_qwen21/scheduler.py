# Qwen-Image 2.1 support for Forge Neo
# two schedules with the pipeline's resolution-dependent time shift, mu = 0.5 @ 256 tokens .. 0.9 @ 8192 tokens
#
# "Viggle Turbo": the timesteps the Viggle turbo LoRAs are distilled on.
# https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo
# 6 steps are the official nodes 1, 0.9375, 0.875, 0.75, 0.5, 0.25; other step counts add or remove nodes at the
# high-noise end only (1 .. 0.875), keeping 0.75, 0.5, 0.25, as Viggle's README asks. Made for turbo checkpoints:
# the last step jumps from sigma ~0.3 to 0, which a base checkpoint cannot do cleanly at any step count.
#
# "Qwen Base": the official diffusers schedule for base checkpoints (QwenImage21Pipeline + its scheduler config):
# linspace(1, 1/n, n), exponential time shift, then stretched so the last sigma before 0 is shift_terminal (0.02)

import math

import torch

NAME = "viggle_turbo"
LABEL = "Viggle Turbo"
# earlier name, so the infotext of images made with it still pastes back
ALIASES = ["Qwen 2.1 Turbo", "qwen21_turbo"]

BASE_NAME = "qwen_base"
BASE_LABEL = "Qwen Base"

LOW_NOISE = (0.75, 0.5, 0.25)
HIGH_NOISE = (1.0, 0.875)

# dynamic shift and terminal of the Qwen-Image 2.1 scheduler config
BASE_SEQ_LEN, MAX_SEQ_LEN = 256, 8192
BASE_SHIFT, MAX_SHIFT = 0.5, 0.9
SHIFT_TERMINAL = 0.02
PATCH = 16


def mu(width: int, height: int) -> float:
    # linear in the token count and not clamped above 8192 tokens, as in diffusers' calculate_shift
    tokens = (width // PATCH) * (height // PATCH)
    return BASE_SHIFT + (MAX_SHIFT - BASE_SHIFT) * (tokens - BASE_SEQ_LEN) / (MAX_SEQ_LEN - BASE_SEQ_LEN)


def _shift(t: list[float], width: int, height: int) -> list[float]:
    e = math.exp(mu(width, height))
    return [e / (e + 1.0 / x - 1.0) for x in t]


def timesteps(steps: int) -> list[float]:
    if steps < len(LOW_NOISE) + 1:
        # too few steps for the turbo nodes: uniform
        return torch.linspace(1.0, 0.0, steps + 1)[:-1].tolist()
    high = torch.linspace(HIGH_NOISE[0], HIGH_NOISE[1], steps - len(LOW_NOISE)).tolist()
    return high + list(LOW_NOISE)


def sigmas(steps: int, width: int, height: int) -> torch.Tensor:
    return torch.FloatTensor(_shift(timesteps(steps), width, height) + [0.0])


def base_sigmas(steps: int, width: int, height: int) -> torch.Tensor:
    shifted = _shift(torch.linspace(1.0, 1.0 / steps, steps).tolist(), width, height)
    # stretch_shift_to_terminal: 1 - s is rescaled so the last value lands on 1 - SHIFT_TERMINAL
    one_minus = [1.0 - s for s in shifted]
    if one_minus[-1] > 0:
        scale = one_minus[-1] / (1.0 - SHIFT_TERMINAL)
        shifted = [1.0 - x / scale for x in one_minus]
    return torch.FloatTensor(shifted + [0.0])


SCHEDULES = {
    LABEL: (NAME, sigmas, ALIASES),
    BASE_LABEL: (BASE_NAME, base_sigmas, []),
}


def _fallback(function):
    def default(n, sigma_min=None, sigma_max=None, device="cpu", **kwargs):
        # used only when the script cannot pass the resolution: 1024x1024
        return function(n, 1024, 1024).to(device)

    return default


def register() -> None:
    from modules import sd_schedulers, shared

    for label, (name, function, aliases) in SCHEDULES.items():
        if label in sd_schedulers.schedulers_map:
            continue
        entry = sd_schedulers.Scheduler(name, label, _fallback(function), aliases=list(aliases))
        sd_schedulers.all_schedulers.append(entry)
        if label not in shared.opts.hide_schedulers:
            sd_schedulers.schedulers.append(entry)
            for key in (name, label, *aliases):
                sd_schedulers.schedulers_map[key] = entry


def _selected(p):
    chosen = getattr(p, "scheduler", None)
    for label, (name, function, aliases) in SCHEDULES.items():
        if chosen in (name, label, *aliases):
            return function
    return None


def is_selected(p) -> bool:
    return _selected(p) is not None


def override(p):
    # Forge passes the resolution only to its Flux2 scheduler; sampler_noise_scheduler_override is the public way in
    function = _selected(p)

    def schedule(steps: int) -> torch.Tensor:
        if getattr(p, "is_hr_pass", False):
            return function(steps, p.hr_upscale_to_x, p.hr_upscale_to_y)
        return function(steps, p.width, p.height)

    return schedule
