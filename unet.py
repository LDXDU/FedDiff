import copy
from abc import abstractmethod
from FR import FR
import numpy as np
import torch as th
from fp16_util import convert_module_to_f16, convert_module_to_f32
from nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from einops import rearrange, reduce


# helpers functions

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
            self,
            spacial_dim: int,
            embed_dim: int,
            num_heads_channels: int,
            output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class stepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class EmbedSequential(nn.Sequential, stepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class BasicResBlock(nn.Module):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # self.emb_layers = nn.Sequential(
        #     nn.SiLU(),
        #     linear(
        #         emb_channels,
        #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        #     ),
        # )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x,), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # emb_out = self.emb_layers(emb).type(h.dtype)
        # while len(emb_out.shape) < len(h.shape):
        #     emb_out = emb_out[..., None]
        # if self.use_scale_shift_norm:
        #     out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        #     scale, shift = th.chunk(emb_out, 2, dim=1)
        #     h = out_norm(h) * (1 + scale) + shift
        #     h = out_rest(h)
        # else:
        #     h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        # print(x.de)
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, num_res_blocks=2):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.init_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.block1_all = []
        self.block2_all = []
        self.res_conv_all = []
        for i in range(num_res_blocks):
            self.block1_all.append(Block(dim, dim_out, groups=groups) if
                                   i == 0 else
                                   Block(dim_out, dim_out, groups=groups))
            self.block2_all.append(Block(dim_out, dim_out, groups=groups))
            self.res_conv_all.append(nn.Conv2d(dim, dim_out, 1) if dim != dim_out and i == 0 else nn.Identity())
        self.block1_all = nn.ModuleList([*self.block1_all])
        self.block2_all = nn.ModuleList([*self.block2_all])
        self.res_conv_all = nn.ModuleList([*self.res_conv_all])

    def _forward(self, x, time_emb=None, i=0):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1_all[i](x, scale_shift=scale_shift)

        h = self.block2_all[i](h)

        return h + self.res_conv_all[i](x)

    def forward(self, x, time_emb=None):
        for i in range(self.num_res_blocks):
            x = self._forward(x, time_emb, i)
        return x


class Conditioning(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Conv2d(dim, dim, 1)

        self.norm_input = LayerNorm(dim, bias=True)
        self.norm_condition = LayerNorm(dim, bias=True)

        self.block = ResnetBlock(dim, dim)

        self.conv2d = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        dtype = x.dtype
        x = fft2(x).type(dtype)
        x = self.ff_parser_attn_map(x)
        x = ifft2(x).real
        x = x.type(dtype)

        x = self.conv2d(x)

        return self.block(x)


class UNetModel(nn.Module):

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        # self.fusion = MF()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self.cond = nn.ModuleList(
            [EmbedSequential(nn.Identity())]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                cond = [
                    Conditioning(int(mult * model_channels))
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                    cond.append(
                        Conditioning(ch)
                    )
                self.cond.append(EmbedSequential(*cond))
                self.input_blocks.append(TimestepEmbedSequential(*layers))

                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                self.cond.append(EmbedSequential(Conditioning(out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.cond.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.cond.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @torch.no_grad()
    def get_feature(self, x, timesteps, y=None):
        # x = torch.squeeze(x, 1)
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module, cond in zip(self.input_blocks, self.cond):
            h = module(h, emb)
            hs.append(h)
            h = cond(h)
        h = self.middle_block(h, emb)
        return h

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # x = torch.squeeze(x, 1)
        # print(x.shape)
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        # print(h.shape)
        for (module, cond) in zip(self.input_blocks, self.cond):
            h = module(h, emb)
            hs.append(h)
            h = cond(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            # print(h.shape)
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return torch.unsqueeze(self.out(h), 1)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    def __init__(self, channel):
        super(EncoderUNetModel, self).__init__()

        self.layer1_x1 = nn.Sequential(
            nn.Conv2d(144, 16, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(16, momentum=0.9),  # note 默认可以修改
            nn.ReLU()
        )

        self.layer1_x1_x1 = nn.Sequential(
            nn.Conv2d(channel, 512, kernel_size=3, stride=1, padding=1),  # todo: paddint
            nn.BatchNorm2d(512, momentum=0.9),  # note 默认可以修改
            nn.ReLU()
        )

        self.layer1_x2 = nn.Sequential(
            nn.Conv2d(21, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=0.9),
            nn.ReLU()
        )

        self.layer1_x2_x2 = nn.Sequential(
            nn.Conv2d(channel, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU()
        )

        self.layer2_x1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # todo
            nn.ReLU()
        )

        self.layer2_x1_x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer2_x2_x2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer2_x2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.layer3_x1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU()
        )
        self.layer3_x2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU()
        )

        self.common_conv1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.common_conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)

        self.layer4_x1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        self.layer4_x2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.layer4_x12 = nn.Sequential(
            nn.Conv2d(64 + 256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        self.layer4_x21 = nn.Sequential(
            nn.Conv2d(64 + 256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.fusion_conv1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.FR1 = FR(128, 128)
        self.FR2 = FR(128, 128)
        self.FR3 = FR(128, 128)
        self.fusion1_layer1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU()
        )

        self.fusion2_layer1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU()
        )
        self.fusion3_layer1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU()
        )

        self.fusion_conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)

        self.fusion1_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            # self.fusion_conv2,
            nn.BatchNorm2d(64, momentum=0.9),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.fusion2_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            # self.fusion_conv2,
            nn.BatchNorm2d(64, momentum=0.9),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.fusion3_layer2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            # self.fusion_conv2,
            nn.BatchNorm2d(64, momentum=0.9),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

        self.fusion_conv3 = nn.Conv2d(64, 15, kernel_size=1, stride=1, padding=0)

        self.fusion1_layer3 = nn.Sequential(
            nn.Conv2d(64, 15, kernel_size=1, stride=1, padding=0)
            # self.fusion_conv3
        )
        self.fusion2_layer3 = nn.Sequential(
            nn.Conv2d(64, 15, kernel_size=1, stride=1, padding=0)
            # self.fusion_conv3
        )
        self.fusion3_layer3 = nn.Sequential(
            nn.Conv2d(64, 15, kernel_size=1, stride=1, padding=0)
            # self.fusion_conv3
        )

    def forward(self, x1, x1_feature, x2, x2_feature):
        if isinstance(x1, np.ndarray):
            x1 = torch.from_numpy(x1)
            x2 = torch.from_numpy(x2)
        x1 = torch.reshape(x1, [-1, 144, 8, 8]).to(torch.float32)
        x2 = torch.reshape(x2, [-1, 21, 8, 8]).to(torch.float32)

        out1_1 = self.layer1_x1(x1)
        out1_2 = self.layer2_x1(out1_1)
        out1_3 = self.layer3_x1(out1_2)
        out1_4 = self.layer4_x1(out1_3)

        out1_1_1 = self.layer1_x1_x1(x1_feature)
        out1_2_1 = self.layer2_x1_x1(out1_1_1)

        out2_1 = self.layer1_x2(x2)
        out2_2 = self.layer2_x2(out2_1)
        out2_3 = self.layer3_x2(out2_2)
        out2_4 = self.layer4_x2(out2_3)

        out2_1_1 = self.layer1_x2_x2(x2_feature)
        out2_2_1 = self.layer2_x2_x2(out2_1_1)
        out1_3 = torch.cat([out1_3, out1_2_1], dim=1)
        out2_3 = torch.cat([out2_3, out2_2_1], dim=1)

        x12 = self.layer4_x12(out1_3)
        x21 = self.layer4_x21(out2_3)
        j1 = self.FR1(out1_4 + x21, out2_4 + x12)
        # print(j1.shape)
        j2 = self.FR2(out1_4, x12)
        j3 = self.FR3(x21, out2_4)
        # j1 = torch.cat([out1_4 + x21, out2_4 + x12], 1)
        # j2 = torch.cat([out1_4, x12], 1)
        # j3 = torch.cat([x21, out2_4], 1)
        # print(j1.shape).
        f11 = self.fusion1_layer1(j1)
        f21 = self.fusion2_layer1(j2)
        f31 = self.fusion3_layer1(j3)

        f12 = self.fusion1_layer2(f11)
        f22 = self.fusion2_layer2(f21)
        f32 = self.fusion3_layer2(f31)

        f13 = self.fusion1_layer3(f12)
        f23 = self.fusion2_layer3(f22)
        f33 = self.fusion3_layer3(f32)

        s1 = f13.shape
        o1 = torch.reshape(f13, [-1, s1[1] * s1[2] * s1[3]])
        s2 = f23.shape
        o2 = torch.reshape(f23, [-1, s2[1] * s2[2] * s2[3]])
        s3 = f33.shape
        o3 = torch.reshape(f33, [-1, s3[1] * s3[2] * s3[3]])
        l2_loss = torch.tensor(0)
        for layer in self.named_parameters():
            if isinstance(layer, nn.Conv2d) and 'bias' not in layer[0]:
                l2_loss += torch.norm(layer[1])
        return o1, o2, o3, l2_loss


if __name__ == "__main__":
    patch_size = 8
    spe = 1
    model = EncoderUNetModel(2560)
    i = torch.randn((1, 144, 8, 8))
    i1 = torch.randn((1, 21, 8, 8))

    i_1 = torch.randn((1, 2560, 4, 4))
    i1_1 = torch.randn((1, 2560, 4, 4))
    print(model(i, i_1, i1, i1_1)[0].shape)
