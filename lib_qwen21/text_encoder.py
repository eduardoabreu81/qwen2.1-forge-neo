# Qwen-Image 2.1 support for Forge Neo
# Qwen3-VL 8B: Forge Neo's Qwen3VL is pinned to the 4B config (Krea 2), this subclass picks the size from the config

from dataclasses import asdict, dataclass

import torch.nn as nn

from backend.nn.llm.llama import Llama2_, Qwen3VL, Qwen3VL_4BConfig
from backend.nn.llm.qwen35 import QWEN3VL_VISION, Qwen3VLVisionModel

VISION_KEYS = ("hidden_size", "intermediate_size", "depth", "num_heads", "num_position_embeddings", "deepstack_visual_indexes")


@dataclass
class Qwen3VL_8BConfig(Qwen3VL_4BConfig):
    hidden_size: int = 4096
    intermediate_size: int = 12288


class Qwen3VL8B(Qwen3VL):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        text_config: dict = config_dict.get("text_config", {})
        config = Qwen3VL_8BConfig() if text_config.get("hidden_size", None) == 4096 else Qwen3VL_4BConfig()

        for key, value in asdict(config).items():
            if key in config_dict:
                assert value == config_dict[key]

        self.num_layers = config.num_hidden_layers
        self.model = Llama2_(config)

        vision_config = {**QWEN3VL_VISION, "out_hidden_size": config.hidden_size}
        for key in VISION_KEYS:
            if key in config_dict.get("vision_config", {}):
                vision_config[key] = config_dict["vision_config"][key]
        self.visual = Qwen3VLVisionModel(vision_config)
