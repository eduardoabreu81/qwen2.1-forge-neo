# Qwen-Image 2.1 support for Forge Neo
# entry point: Forge Neo imports this at startup, before the UI is built and before any model is loaded

import traceback

import gradio as gr

import modules.scripts as scripts

from lib_qwen21 import compat

ENABLED = False
INFOTEXT_TRANSPARENT = "Transparent background"

_problems = compat.check()

if _problems:
    compat.report(_problems)
else:
    try:
        from lib_qwen21 import patches

        patches.apply()
        ENABLED = True
    except Exception:
        print("[Qwen-Image 2.1] failed to enable, Forge Neo is unchanged:")
        traceback.print_exc()


_warned_sigma = False


def _warn_discarded_sigma(p) -> None:
    # a user setting, left as it is: the extension only says what it costs a few-step model
    global _warned_sigma
    from modules import shared

    discard = p.override_settings.get("always_discard_next_to_last_sigma", getattr(shared.opts, "always_discard_next_to_last_sigma", False))
    if discard and not _warned_sigma:
        _warned_sigma = True
        print("[Qwen-Image 2.1] 'Always discard next-to-last sigma' is on: few-step (turbo) generations lose fine detail. It is under Settings > Sampler Parameters (visible with --adv-samplers).")


class Qwen21Script(scripts.Script):
    # options for Qwen-Image 2.1; with any other model the script does nothing
    def title(self):
        return "Qwen-Image 2.1"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else False

    def ui(self, is_img2img):
        with gr.Accordion("Qwen-Image 2.1", open=False):
            transparent = gr.Checkbox(label="Transparent background (RGBA PNG)", value=False, elem_id=self.elem_id("transparent"))
        self.infotext_fields = [(transparent, INFOTEXT_TRANSPARENT)]
        return [transparent]

    def before_process(self, p, *args):
        # a module that imported the LoRA key mapping after startup still holds the original
        from lib_qwen21 import lora

        lora.hook()

    def process(self, p, transparent=False, *args):
        from lib_qwen21 import alpha, prompting

        if not alpha.is_qwen21(p):
            return
        _warn_discarded_sigma(p)
        if not transparent:
            return
        p.all_prompts = [prompting.transparent(x) for x in p.all_prompts]
        if getattr(p, "all_hr_prompts", None):
            p.all_hr_prompts = [prompting.transparent(x) for x in p.all_hr_prompts]
        p.extra_generation_params[INFOTEXT_TRANSPARENT] = True

    def postprocess_image(self, p, pp, *args):
        from lib_qwen21 import alpha

        pp.image = alpha.apply(p, pp.image)
