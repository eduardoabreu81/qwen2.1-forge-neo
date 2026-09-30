# Qwen-Image 2.1 support for Forge Neo
# ported from https://github.com/Comfy-Org/ComfyUI/blob/6bfaacc67c2103481e5f0c84d75257cd0581d86a/comfy/text_encoders/qwen_image21.py

import numbers

import torch

from backend.args import dynamic_args
from backend.text_processing import emphasis

from backend.text_processing._comfy import EMBEDDINGS, INF, TOKEN_WEIGHTS, SDClipModel, SDTokenizer

IM_START = 151644
IMAGE_PAD = 151655


class Qwen3VL8BEngine:
    def __init__(self, text_encoder, tokenizer):
        # last layer without the final RMSNorm: transformers 4.57 hidden_states[-1], which Qwen's results are tuned to
        self.text_encoder = SDClipModel(text_encoder, layer="hidden", layer_idx=-1, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=INF, min_length=1, pad_token=151643)

        # Qwen-Image 2.1 keeps the reasoning turn: no empty <think> block is appended
        self.llama_template = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> EMBEDDINGS | list[EMBEDDINGS]:
        return self.tokenizer.tokenizer(texts)["input_ids"]

    def __call__(self, texts: list[str], images: list[torch.Tensor] = []) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}

        for line in texts:
            if line in cache:
                cond = cache[line]
            else:
                chunk = self._tokenize_with_weights(line, images)
                cond = self.text_encoder.encode_token_weights(chunk)[0]
                cond = cond[:, self._system_turn_length(chunk[0]) :]
                cache[line] = cond

            zs.extend(cond)  # (L, D) per prompt; the prompt parser stacks the batch

        return zs

    @staticmethod
    def _system_turn_length(tok_pairs: list[tuple]) -> int:
        # drop the system turn: everything before the second <|im_start|>
        count_im_start = 0
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral) and elem == IM_START:
                count_im_start += 1
                if count_im_start == 2:
                    return i
        return 0

    def _tokenize_with_weights(self, text: str, images: list[torch.Tensor]) -> TOKEN_WEIGHTS:
        refs = " ".join(f"<image{i + 1}>{self.vision_block}" for i in range(len(images)))
        llama_text = self.llama_template.format(refs + text.strip())
        tokens = self.tokenizer.tokenize_with_weights(llama_text, disable_weights=True)

        embed_count = 0

        for r in tokens:
            for i in range(len(r)):
                if isinstance(r[i][0], (int, float)) and r[i][0] == IMAGE_PAD:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count], "original_type": "image"},) + r[i][1:]
                        embed_count += 1

        return tokens
