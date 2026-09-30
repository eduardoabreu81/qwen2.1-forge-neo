# Qwen-Image 2.1 support for Forge Neo
# entry point: Forge Neo imports this at startup, before the UI is built and before any model is loaded

import traceback

import modules.scripts as scripts

from lib_qwen21 import compat

ENABLED = False

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


class Qwen21Transparency(scripts.Script):
    # no UI: attaches the alpha channel of Qwen-Image 2.1 images, anything else passes through
    def title(self):
        return "Qwen-Image 2.1 transparency"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ENABLED else False

    def postprocess_image(self, p, pp, *args):
        from lib_qwen21 import alpha

        pp.image = alpha.apply(p, pp.image)
