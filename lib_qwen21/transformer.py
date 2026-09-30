# Qwen-Image 2.1 support for Forge Neo
# ported from https://github.com/Comfy-Org/ComfyUI/blob/6bfaacc67c2103481e5f0c84d75257cd0581d86a/comfy/ldm/qwen_image21/model.py
# https://github.com/huggingface/diffusers (Apache 2.0) Qwen-Image 2.1

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.attention import attention_function
from backend.memory_management import cast_to
from backend.nn.flux import EmbedND, timestep_embedding
from backend.quant_ops import ck


class ZeroCenteredRMSNorm(nn.Module):
    # stored weight is scale - 1, applied in fp32
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = cast_to(self.weight, dtype=torch.float32, device=x.device) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = cast_to(self.weight, dtype=torch.float32, device=x.device)
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(in_dim, eps=eps)
        self.in_layer = nn.Linear(in_dim, hidden_size, bias=False)
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=False)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.timestep_embedder(timestep_embedding(timestep.float(), 256).to(dtype))


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, fused: bool = True):
        super().__init__()
        self.fused = fused
        if fused:
            # [gate; up] in one GEMM
            self.gate_up = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.proj = nn.Linear(dim, hidden_dim, bias=False)
            self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            return self.out(F.silu(gate).mul_(up))
        return self.out(F.silu(self.gate_layer(x)).mul_(self.proj(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-6):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False)])
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, prefix_len: int, transformer_options={}) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, -1).transpose(1, 2)
        k = self.to_k(x).view(B, N, self.heads, -1).transpose(1, 2)
        v = self.to_v(x).view(B, N, self.heads, -1).transpose(1, 2)
        q, k = self.norm_q(q), self.norm_k(k)
        q, k = ck.apply_rope(q, k, pe)
        return self.to_out[0](block_causal_attention(q, k, v, self.heads, prefix_len, transformer_options))


def block_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, prefix_len: int, transformer_options={}) -> torch.Tensor:
    # (B, H, N, D); the text prefix attends causally to itself, the image attends to everything
    B, H, N, D = q.shape
    image = attention_function(q[:, :, prefix_len:], k, v, heads, skip_reshape=True, transformer_options=transformer_options)
    if prefix_len == 0:
        return image
    text = F.scaled_dot_product_attention(q[:, :, :prefix_len], k[:, :, :prefix_len], v[:, :, :prefix_len], is_causal=True)
    text = text.transpose(1, 2).reshape(B, prefix_len, H * D)
    return torch.cat((text, image), dim=1)


def _split_rows(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # shared modulation rows: (t = 0 row for text, sampled-t rows for the image)
    return p[-1:].unsqueeze(1), p[:-1].unsqueeze(1)


def _modulated_norm(norm: nn.LayerNorm, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor], prefix_len: int) -> torch.Tensor:
    s_prefix, s_target = scale
    out = norm(x)
    return torch.cat((out[:, :prefix_len] * (1 + s_prefix), out[:, prefix_len:] * (1 + s_target)), dim=1)


def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: tuple[torch.Tensor, torch.Tensor], prefix_len: int) -> torch.Tensor:
    g_prefix, g_target = gate
    return torch.cat((torch.addcmul(x[:, :prefix_len], y[:, :prefix_len], g_prefix), torch.addcmul(x[:, prefix_len:], y[:, prefix_len:], g_target)), dim=1)


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: int = 3, eps: float = 1e-6, fused_mlp: bool = True):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, num_attention_heads, attention_head_dim, eps=eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLUFeedForward(dim, dim * mlp_ratio, fused=fused_mlp)

    def forward(self, x: torch.Tensor, mod: tuple, pe: torch.Tensor, prefix_len: int, transformer_options={}) -> torch.Tensor:
        scale1, gate1, scale2, gate2 = mod
        x = _gated_residual(x, self.attn(_modulated_norm(self.img_norm1, x, scale1, prefix_len), pe, prefix_len, transformer_options), gate1, prefix_len)
        x = _gated_residual(x, self.img_mlp(_modulated_norm(self.img_norm2, x, scale2, prefix_len)), gate2, prefix_len)
        if x.dtype == torch.float16:
            x = x.clip(-65504, 65504)
        return x


class LastLayer(nn.Module):
    # scale only, no shift
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale = self.linear(F.silu(temb)).unsqueeze(1)
        return self.norm(x) * (1 + scale)


class QwenImage21Transformer2DModel(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        fused_mlp=True,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps=eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)

        # one modulation shared by every block
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))

        self.transformer_blocks = nn.ModuleList([QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, eps=eps, fused_mlp=fused_mlp) for _ in range(num_layers)])

        self.norm_out = LastLayer(self.inner_dim, eps=eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)

    def build_sequence(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        # text first, target image last
        txt = self.txt_in(context)
        n = txt.shape[1]
        h, w = x.shape[-2:]
        device = x.device

        txt_ids = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1).expand(n, 3)
        hh = torch.arange(h, device=device, dtype=torch.float32) - (h - h // 2)
        ww = torch.arange(w, device=device, dtype=torch.float32) - (w - w // 2)
        img_ids = torch.stack((torch.full((h, w), n, device=device, dtype=torch.float32), hh[:, None].expand(h, w), ww[None, :].expand(h, w)), dim=-1).flatten(0, 1)

        img = self.img_in(x.flatten(2).transpose(1, 2))
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=0).unsqueeze(0))
        return torch.cat((txt, img), dim=1), pe, n

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor, transformer_options={}, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        dtype = x.dtype

        hidden_states, pe, prefix_len = self.build_sequence(x, context.to(dtype))

        # the pipeline rounds t*1000 and t to the compute dtype; text tokens modulate from t = 0
        t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
        temb = self.time_text_embed(torch.cat((t, t.new_zeros(1))), dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2), _split_rows(gate2.tanh()))

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, mod, pe, prefix_len, transformer_options=transformer_options)

        hidden_states = self.norm_out(hidden_states[:, prefix_len:], temb[:-1])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states.transpose(1, 2).reshape(B, self.out_channels, H, W)
