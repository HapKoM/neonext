import torch
import torch.nn as nn
from ptflops.pytorch_ops import linear_flops_counter_hook

from ptvision.models.neonet_utils import NeoCellCPP

TORCH_MAJOR, TORCH_MINOR = list(map(int, torch.__version__.split('.')[:2]))
if TORCH_MAJOR >=1 and TORCH_MINOR >= 9:
    LinearWithBias = nn.modules.linear.NonDynamicallyQuantizableLinear
else:
    LinearWithBias = nn.modules.linear._LinearWithBias


def identity_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def neocell_real_flops_counter_hook(module, input, output):
    flops = 0

    batch_size, x_cin, x_hin, x_win = input[0].size()
    h_a, w_a = module.a_full_height, module.a_full_width
    h_b, w_b = module.b_full_height, module.b_full_width

    channels = 0
    for channel_spec in module.channel_specs:
        channels += channel_spec["channels"]

    # forward
    # matrix A
    flops += channels * h_a * w_a * h_b
    # matrix B
    flops += channels * h_a * h_b * w_b

    # # backward
    # # recover Ax
    # flops += channels * h_a * w_a * h_b
    # # grad_B
    # flops += channels * h_b * h_a * w_b
    # flops += channels * h_b * w_b # avg gradients over batch_size
    # # grad_Ax
    # flops += channels * h_a * w_b * h_b
    # # grad_A
    # flops += channels * h_a * h_b * w_a
    # flops += channels * h_a * w_a # avg gradients over batch_size
    # # grad_x
    # flops += channels * w_a * h_a * h_b

    flops *= batch_size
    module.__flops__ += flops


def layernorm_flops_counter_hook(module, input, output):
    input = input[0]

    flops = input.numel()
    if (getattr(module, 'affine', False) or
        getattr(module, 'elementwise_affine', False)):
        flops *= 2
    module.__flops__ += flops


CUSTOM_MODULES_HOOKS = {
    nn.Identity: identity_flops_counter_hook,
    nn.LayerNorm: layernorm_flops_counter_hook,
    NeoCellCPP: neocell_real_flops_counter_hook,
    LinearWithBias: linear_flops_counter_hook
}
