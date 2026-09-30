# Qwen-Image 2.1 support for Forge Neo
# ported from https://github.com/Comfy-Org/ComfyUI/blob/6bfaacc67c2103481e5f0c84d75257cd0581d86a/comfy/ldm/wan/vae2_2.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Qwen-Image 2.1 VAE: Wan 2.2 layout with temporal kernel 1, no patchify, RGBA

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config

from backend.nn._vae import ProcessLatent
from backend.nn.wan_vae import AttentionBlock, RMS_norm
from backend.nn.wan_vae import CausalConv3d as _CausalConv3d
from backend.operations import get_weight_and_bias, main_stream_worker, weights_manual_cast

STRIP_ELEMS = 2**24


class CausalConv3d(_CausalConv3d):
    # the temporal kernel is 1 everywhere: on a single frame this is exactly a Conv2d over weight[:, :, 0],
    # and cuDNN's 3d path is far slower than the 2d one (a stalled decode on Turing)

    def forward(self, x, *args, **kwargs):
        if self.kernel_size[0] != 1 or x.shape[2] != 1 or self.stride[0] != 1:
            return super().forward(x, *args, **kwargs)
        if self.parameters_manual_cast:
            weight, bias, signal = weights_manual_cast(self, x)
            with main_stream_worker(weight, bias, signal):
                return self._conv2d(x, weight, bias)
        weight, bias = get_weight_and_bias(self)
        return self._conv2d(x, weight, bias)

    def _conv2d(self, x, weight, bias):
        return F.conv2d(x.squeeze(2), weight.squeeze(2), bias, self.stride[1:], self.padding[1:], self.dilation[1:], self.groups).unsqueeze(2)


def strip_apply(fn, x, scale=1, halo=1, out=None):
    # strips of rows bound cudnn's conv workspace, a halo row per 3x3 conv keeps them exact
    n = -(-x.numel() * scale * scale // STRIP_ELEMS)
    if n <= 1 and out is None:
        return fn(x)
    add = out is not None
    size = x.shape[-2]
    step = -(-size // n)
    for a in range(0, size, step):
        b = min(size, a + step)
        lo = max(0, a - halo)
        y = fn(x.narrow(-2, lo, min(size, b + halo) - lo)).narrow(-2, (a - lo) * scale, (b - a) * scale)
        if out is None:
            out = y.new_empty(*y.shape[:-2], size * scale, y.shape[-1])
        dst = out.narrow(-2, a * scale, (b - a) * scale)
        if add:
            dst.add_(y)
        else:
            dst.copy_(y)
    return out


def conv3x3(in_dim, out_dim, temporal_kernel=1):
    return CausalConv3d(in_dim, out_dim, (temporal_kernel, 3, 3), padding=(temporal_kernel // 2, 1, 1))


class Resample(nn.Module):
    # single image only: the temporal convolutions of the 3d modes never run without a feature cache

    def __init__(self, dim, mode, temporal_kernel=1):
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (temporal_kernel, 1, 1), padding=(temporal_kernel // 2, 0, 0))
        else:
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (temporal_kernel, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if self.mode.startswith("upsample"):
            x = strip_apply(self.resample, x, scale=2)
        else:
            x = self.resample(x)
        return x.reshape(b, t, c, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0, temporal_kernel=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            conv3x3(in_dim, out_dim, temporal_kernel),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            conv3x3(out_dim, out_dim, temporal_kernel),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        # the whole block runs in strips so its intermediates never exist at full size
        return strip_apply(lambda s: self.residual(s).add_(self.shortcut(s)), x, halo=2)


class AvgDown3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return x.mean(dim=2)


class DupUp3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(x.size(0), self.out_channels, self.factor_t, self.factor_s, self.factor_s, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t, x.size(4) * self.factor_s, x.size(6) * self.factor_s)
        # first chunk: only the last frame of the temporal expansion is real
        return x[:, :, self.factor_t - 1 :, :, :]


class Down_ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False, temporal_kernel=1):
        super().__init__()

        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)

        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout, temporal_kernel=temporal_kernel))
            in_dim = out_dim

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode, temporal_kernel=temporal_kernel))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x):
        return self.downsamples(x) + self.avg_shortcut(x)


class Up_ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, mult, temperal_upsample=False, up_flag=False, temporal_kernel=1):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
        else:
            self.avg_shortcut = None

        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout, temporal_kernel=temporal_kernel))
            in_dim = out_dim

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode, temporal_kernel=temporal_kernel))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x):
        x_main = self.upsamples(x)
        if self.avg_shortcut is None:
            return x_main
        return strip_apply(self.avg_shortcut, x, scale=self.avg_shortcut.factor_s, halo=0, out=x_main)


class Encoder3d(nn.Module):

    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout, in_channels, temporal_kernel):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]

        self.conv1 = conv3x3(in_channels, dims[0], temporal_kernel)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(Down_ResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks, temperal_downsample=t_down_flag, down_flag=i != len(dim_mult) - 1, temporal_kernel=temporal_kernel))
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout, temporal_kernel=temporal_kernel),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout, temporal_kernel=temporal_kernel),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            conv3x3(out_dim, z_dim, temporal_kernel),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.downsamples(x)
        x = self.middle(x)
        return self.head(x)


class Decoder3d(nn.Module):

    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, attn_scales, temperal_upsample, dropout, out_channels, temporal_kernel):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv1 = conv3x3(z_dim, dims[0], temporal_kernel)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout, temporal_kernel=temporal_kernel),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout, temporal_kernel=temporal_kernel),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(Up_ResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks + 1, temperal_upsample=t_up_flag, up_flag=i != len(dim_mult) - 1, temporal_kernel=temporal_kernel))
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            conv3x3(out_dim, out_channels, temporal_kernel),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.middle(x)
        x = self.upsamples(x)
        return self.head(x)


class QwenImage21VAE(nn.Module, ProcessLatent, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        base_dim=96,
        decoder_base_dim=144,
        z_dim=64,
        dim_mult=[1, 2, 4, 8, 8],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        dropout=0.0,
        in_channels=4,
        out_channels=4,
        temporal_kernel=1,
        latent_channels=64,
    ):
        # latent_channels mirrors z_dim: the stock Forge VAE wrapper reads it from the config
        super().__init__()
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.temperal_upsample = temperal_downsample[::-1]
        self.alpha_target_hw: tuple[int, int] | None = None
        self.alpha_chunks: list[torch.Tensor] = []

        self.encoder = Encoder3d(base_dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temperal_downsample, dropout, in_channels, temporal_kernel)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(decoder_base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout, out_channels, temporal_kernel)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) in [-1, 1]; RGB input gets an opaque alpha channel
        if x.shape[1] < self.in_channels:
            x = F.pad(x, (0, 0, 0, 0, 0, self.in_channels - x.shape[1]), value=1.0)
        return self.conv1(self.encoder(x.unsqueeze(2))).chunk(2, dim=1)[0].squeeze(2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # (B, 64, H/16, W/16) -> (B, 3, H, W) in [-1, 1]; the webui carries RGB only, so the alpha channel of a
        # whole-image decode is kept aside (alpha_target_hw is set by the engine; tiles never match it)
        out = self.decoder(self.conv2(z.unsqueeze(2))).squeeze(2)
        if out.shape[1] > 3 and self.alpha_target_hw == tuple(out.shape[-2:]):
            self.alpha_chunks.append(out[:, 3:4].detach().to(device="cpu", dtype=torch.float32))
        return out[:, :3]
