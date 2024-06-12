"""
    Implementation of paper "NeoNeXt: Novel neural network operator and architecture
    based on the patch-wise matrix multiplications", https://arxiv.org/abs/2403.11251
"""
from collections import OrderedDict

import torch
from torch import nn

from .neonext_utils import (
    NeoCellCPP, NeoBottleNeck,
    Patchify, GlobalAveragePooling,
    LayerNorm, get_kernel_spec,
    xavier_uniform_init, kaiming_normal_init, neo_init
)


def make_downsample(dims_in, dims_out, neocell_cfg):
    """
        Helper function that creates downsample module
        Args:
            dims_in: Number of input channels
            dims_out: Number of output channels
            neocell_cfg: Configuration dictionary of the NeoCel downsampling block
        Returns:
            Downsampling module
    """
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(dims_in, dims_out, kernel_size=1, bias=False)),
        ('neocell', NeoCellCPP(**neocell_cfg)),
        ('norm_1', nn.BatchNorm2d(dims_out)),
    ]))


def make_stem(dims_out):
    """
        Helper function that creates stem module
        Args:
            dims_out: Number of output channels
        Returns:
            Stem module
    """
    return nn.Sequential(OrderedDict([
        ('patchify', Patchify(4)), # 48x56x56
        ('conv', nn.Conv2d(48, dims_out, kernel_size=1, bias=False)), # dims[0]x56x56
        ('norm', nn.BatchNorm2d(dims_out))
    ]))


def make_stage(n_blocks, kernel_spec, n_channels, shift,
               dp_rates, layer_scale_init_value, linear_bias):
    """
        Helper function that creates NeoNeXt stage
        Args:
            n_layers: Number of NeoBottleNeck blocks in the stage
            kernel_spec: kernel specification for this stage,
                         see spec_code parameter of get_kernel_spec function
            n_channels: number of channels between the stage blocks
            shift: Indicates wether NeoCell spatial shifting is used in this stage
            dp_rates: list of the DropPath rates for the blocks in this stage
            layer_scale_init_value: initial value of LayerScale
            linear_bias: Indicates wether bias is enabled in the linear layers
        Returns:
            Sequential container of NeoBottleNeck blocks
    """
    assert len(dp_rates) == n_blocks
    layers = []
    for i in range(n_blocks):
        channels_spec = get_kernel_spec(kernel_spec, n_channels, shift)
        layers.append(
            NeoBottleNeck(
                channel_specs=channels_spec,
                c_outer=n_channels,
                drop_path=dp_rates[i],
                layer_scale_init_value=layer_scale_init_value,
                linear_bias=linear_bias
            )
        )
    return nn.Sequential(*layers)


class NeoNeXt(nn.Module):
    """
    Implementation of NeoNeXt architecture.

    Args:
        num_classes: number of classes that model can classify
        per_stage_layers: tuple of 4 elements containing number of blocks in each of 4 stages
        dims: tuple of 4 elements containing number of channels in each of 4 stages
        drop_path: maximal DropPath value
                   (linearly increased from 0 to drop_path for the residual blocks)
    """
    def __init__(
        self, num_classes=1000, per_stage_layers=(1, 1, 1, 1), dims=(96, 192, 384, 768),
        drop_path=0.0, conv_init=None, shifts=(1, 1, 1, 0),
        layer_scale_init_value=0, linear_bias=False, kernel_spec='default'
    ):
        super().__init__()
        if isinstance(shifts, str):
            shifts = [int(s) for s in shifts.split(",")]
        if isinstance(per_stage_layers, str):
            per_stage_layers = [int(p) for p in per_stage_layers.split(",")]
        print(f"NeoNeXt: per_stage_layers = {per_stage_layers}")
        print(f"NeoNeXt: conv_init = {conv_init}")
        print(f"NeoNeXt: shifts = {shifts}")
        print(f"NeoNeXt: layer_scale_init_value = {layer_scale_init_value}")
        print(f"NeoNeXt: linear_bias = {linear_bias}")
        print(f"NeoNeXt: kernel_spec = {kernel_spec}")

        self.stem = make_stem(dims[0])

        dp_rates = [
            x.item()
            for x in torch.linspace(0, drop_path, sum(per_stage_layers))
        ]
        start_depth = 0

        # Stage 1
        # Default kernel for stage 1 is 4+7
        spec_code = "4+7" if kernel_spec == "default" else kernel_spec
        self.stage_1 = make_stage(
            per_stage_layers[0], spec_code, dims[0],
            shifts[0], dp_rates[start_depth:start_depth+per_stage_layers[0]],
            layer_scale_init_value, linear_bias
        )

        ds1_neocell_cfg = {
            'channel_specs': [{"h_in": 2, "h_out": 1, "w_in": 2, "w_out": 1,
                               "channels": dims[1], "shift": 0}],
        }
        self.downsample_1 = make_downsample(dims[0], dims[1], ds1_neocell_cfg)

        start_depth += per_stage_layers[0]

        # Stage 2
        # Default kernel for stage 2 is 4+7
        #     spec_code = "4+7" if kernel_spec == "default" else kernel_spec
        self.stage_2 = make_stage(
            per_stage_layers[1], spec_code, dims[1],
            shifts[1], dp_rates[start_depth:start_depth+per_stage_layers[1]],
            layer_scale_init_value, linear_bias
        )

        ds2_neocell_cfg = {
            'channel_specs': [{"h_in": 2, "h_out": 1, "w_in": 2, "w_out": 1,
                               "channels": dims[2], "shift": 0}],
        }
        self.downsample_2 = make_downsample(dims[1], dims[2], ds2_neocell_cfg)

        start_depth += per_stage_layers[1]

        # Stage 3
        # Default kernel for stage 3 is 7
        spec_code = "7" if kernel_spec == "default" else kernel_spec
        self.stage_3 = make_stage(
            per_stage_layers[2], spec_code, dims[2],
            shifts[2], dp_rates[start_depth:start_depth+per_stage_layers[2]],
            layer_scale_init_value, linear_bias
        )

        ds3_neocell_cfg = {
            'channel_specs': [{"h_in": 2, "h_out": 1, "w_in": 2, "w_out": 1,
                               "channels": dims[3], "shift": 0}],
        }
        self.downsample_3 = make_downsample(dims[2], dims[3], ds3_neocell_cfg)

        start_depth += per_stage_layers[2]

        # Stage 4
        # Default kernel for stage 4 is 7
        spec_code = "7" if kernel_spec == "default" else kernel_spec
        self.stage_4 = make_stage(
            per_stage_layers[3], spec_code, dims[3],
            shifts[3], dp_rates[start_depth:start_depth+per_stage_layers[3]],
            layer_scale_init_value, linear_bias
        )

        # Head
        self.head = nn.Sequential(OrderedDict([
            ('pool', GlobalAveragePooling()),
            ('ln', LayerNorm(dims[3], eps=1e-6)),
            ('flatten', nn.Flatten()), # dims[3]
            ('fc', nn.Linear(dims[3], num_classes)),
        ]))

        if conv_init:
            if conv_init == 'xavier_uniform':
                self.apply(xavier_uniform_init)
            elif conv_init == 'kaiming_normal':
                self.apply(kaiming_normal_init)
            else:
                raise NotImplementedError(
                    f"Unsupported convolution initialization: {conv_init}"
                )
        self.apply(neo_init)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage_1(out)
        out = self.downsample_1(out)
        out = self.stage_2(out)
        out = self.downsample_2(out)
        out = self.stage_3(out)
        out = self.downsample_3(out)
        out = self.stage_4(out)
        logits = self.head(out)
        return logits


def neonext_t(**kwargs):
    return NeoNeXt(
        per_stage_layers=[3, 3, 9, 3], dims=[96, 192, 384, 768],  **kwargs)


def neonext_s(**kwargs):
    return NeoNeXt(
        per_stage_layers=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


def neonext_b(**kwargs):
    return NeoNeXt(
        per_stage_layers=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


def neonext_l(**kwargs):
    return NeoNeXt(
        per_stage_layers=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


def neonext_xl(**kwargs):
    return NeoNeXt(
        per_stage_layers=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
