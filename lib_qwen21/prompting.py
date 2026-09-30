# Qwen-Image 2.1 support for Forge Neo
# the prompt format Qwen recommends for transparent (RGBA) images

PREFIX = "This is an RGBA image with transparency."
SUFFIX = "The image has alpha channel and the background is transparent."

# a prompt that already asks for transparency is left as written
MARKERS = ("rgba image", "alpha channel")


def transparent(prompt: str) -> str:
    if any(m in prompt.lower() for m in MARKERS):
        return prompt
    body = prompt.strip().rstrip(".").strip()
    body = body[:1].upper() + body[1:]
    return f"{PREFIX} {body}. {SUFFIX}" if body else f"{PREFIX} {SUFFIX}"
