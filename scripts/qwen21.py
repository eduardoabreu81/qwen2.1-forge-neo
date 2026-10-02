# Qwen-Image 2.1 support for Forge Neo
# entry point: Forge Neo imports this at startup, before the UI is built and before any model is loaded

import traceback

import gradio as gr

import modules.scripts as scripts

from lib_qwen21 import compat

ENABLED = False
INFOTEXT_TRANSPARENT = "Transparent background"
INFOTEXT_COMPUTE = "Qwen-Image 2.1 compute"
INFOTEXT_DEGRID = "Qwen-Image 2.1 degrid"

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
            degrid = gr.Checkbox(label="Remove VAE grid (the 2- and 4-pixel pattern of the Qwen-Image 2.1 VAE)", value=True, elem_id=self.elem_id("degrid"))
            controls = [transparent, degrid]
            if ENABLED:
                from lib_qwen21 import reference_ui

                controls += reference_ui.build(self.elem_id, is_img2img)
        self.infotext_fields = [(transparent, INFOTEXT_TRANSPARENT), (degrid, INFOTEXT_DEGRID)]
        return controls

    def before_process(self, p, *args):
        # a module that imported the LoRA key mapping after startup still holds the original
        from lib_qwen21 import lora

        lora.hook()

    def process(self, p, transparent=False, degrid=True, *args):
        from lib_qwen21 import alpha, prompting, reference_ui

        if not alpha.is_qwen21(p):
            return
        _warn_discarded_sigma(p)
        reference_ui.apply(p, *reference_ui.collect(p, args, self.is_img2img))
        from lib_qwen21 import scheduler

        if scheduler.is_selected(p) and p.sampler_noise_scheduler_override is None:
            p.sampler_noise_scheduler_override = scheduler.override(p)
        # the dtype the transformer actually computes in, so two images can be told apart
        compute = getattr(p.sd_model.forge_objects.unet.model, "computation_dtype", None)
        if compute is not None:
            p.extra_generation_params[INFOTEXT_COMPUTE] = str(compute).replace("torch.", "")
        if degrid:
            p.extra_generation_params[INFOTEXT_DEGRID] = True
        if not transparent:
            return
        p.all_prompts = [prompting.transparent(x) for x in p.all_prompts]
        if getattr(p, "all_hr_prompts", None):
            p.all_hr_prompts = [prompting.transparent(x) for x in p.all_hr_prompts]
        p.extra_generation_params[INFOTEXT_TRANSPARENT] = True

    def before_process_init_images(self, p, pp, *args, **kwargs):
        from lib_qwen21 import alpha, reference_ui

        if alpha.is_qwen21(p) and p.sd_model.references:
            reference_ui.crop_to_inpaint_area(p, pp.get("crop_region"))

    def process_before_every_sampling(self, p, *args, **kwargs):
        from lib_qwen21 import alpha, reference

        if not alpha.is_qwen21(p) or not p.sd_model.references:
            return
        noise = p.modified_noise if getattr(p, "modified_noise", None) is not None else kwargs["noise"]
        p.modified_noise = reference.decorrelate(noise, p.sd_model.references)

    def postprocess_image(self, p, pp, transparent=False, degrid=True, *args):
        from lib_qwen21 import alpha

        if degrid and alpha.is_qwen21(p):
            from lib_qwen21 import degrid as grid

            pp.image = grid.remove(pp.image)
        pp.image = alpha.apply(p, pp.image)
