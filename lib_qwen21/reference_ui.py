# Qwen-Image 2.1 support for Forge Neo
# reference images: in img2img the input image is <image1>; the gallery of Forge Neo's own ImageStitch Integrated
# holds the rest, numbered on screen by style.css. In txt2img the gallery holds them all, for a new picture

import gradio as gr

from . import reference

INFOTEXT_REFERENCES = "Qwen-Image 2.1 references"
INFOTEXT_REFERENCE_SIZE = "Qwen-Image 2.1 reference size"
IMAGE_STITCH = "ImageStitch Integrated"
# the model was trained with up to 10 references
MAX_REFERENCES = 10
SIZES = ["1024x1024", "Same as the output"]
# past this many, the references take more memory than the image being made at 1 MP
MANY_REFERENCES = 3

_warned_many = False
_warned_max = False


def build(elem_id, is_img2img: bool) -> list:
    controls = []
    if is_img2img:
        controls.append(gr.Checkbox(label="Use the input image as reference (editing)", value=False, elem_id=elem_id("reference")))
        gr.Markdown(f"The input image is `<image1>`. For more references, up to 10 in all, add them to the **{IMAGE_STITCH}** gallery: they are `<image2>`, `<image3>`… in the order shown.", elem_classes="qwen21-note")
    else:
        gr.Markdown(f"To make a new picture from reference images, up to 10, add them to the **{IMAGE_STITCH}** gallery: they are `<image1>`, `<image2>`… in the order shown.", elem_classes="qwen21-note")
    controls.append(gr.Radio(SIZES, value=SIZES[0], label="Reference size", elem_id=elem_id("reference_size")))
    return controls


def _gallery(p) -> list:
    # the images in ImageStitch's gallery, when its accordion is on
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", []):
        if script.title() == IMAGE_STITCH:
            enable, gallery = (list(p.script_args[script.args_from : script.args_to]) + [False, None])[:2]
            if not enable or not gallery:
                return []
            images = [reference.load(item) for item in gallery]
            return [image for image in images if image is not None]
    return []


def collect(p, args, is_img2img: bool) -> tuple[list, str]:
    # the arguments of build(), in order; an API call may leave out the trailing ones
    global _warned_max
    if is_img2img:
        use_reference, size = (*args, *(False, SIZES[0])[len(args) :])[:2]
        if not use_reference:
            return [], size
        images = (getattr(p, "init_images", None) or [])[:1]
    else:
        (size,) = (*args, *(SIZES[0],)[len(args) :])[:1]
        images = []
    images += _gallery(p)
    if len(images) > MAX_REFERENCES:
        if not _warned_max:
            _warned_max = True
            print(f"[Qwen-Image 2.1] {len(images)} reference images: the model takes up to {MAX_REFERENCES}, the rest are left out.")
        images = images[:MAX_REFERENCES]
    return images, size


def apply(p, references: list, size: str) -> None:
    global _warned_many
    engine = p.sd_model
    resolution = round((p.width * p.height) ** 0.5) if size == SIZES[1] else reference.RESOLUTION
    # Forge caches the conditioning by prompt, which knows nothing of the references
    if references or engine.references:
        p.clear_prompt_cache()
    engine.set_references(references, resolution)
    if not references:
        return
    p.extra_generation_params[INFOTEXT_REFERENCES] = len(references)
    if size == SIZES[1]:
        p.extra_generation_params[INFOTEXT_REFERENCE_SIZE] = SIZES[1]
    if len(references) > MANY_REFERENCES and not _warned_many:
        _warned_many = True
        print(f"[Qwen-Image 2.1] {len(references)} reference images: each one adds to every step about what a 1024x1024 image costs. With 8-12 GB of VRAM, keep to 2-3.")


def crop_to_inpaint_area(p, crop_region) -> None:
    # "Only masked" makes the picture from the masked area alone: <image1> has to be that area too,
    # or the model draws the whole input image into it
    engine = p.sd_model
    images, resolution = engine.reference_sources
    if crop_region is None or not images or not getattr(p, "init_images", None):
        return
    images = [p.init_images[0].crop(crop_region), *images[1:]]
    engine.set_references(images, resolution)
