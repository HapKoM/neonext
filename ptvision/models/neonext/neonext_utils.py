import math
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

from ptvision.nvtx_profiler import get_nvtx_profiler
from . import neonext_cpp_module as neocell_cpp

__all__ = [
	'NeoCellCPP', 'NeoBottleNeck', 'Patchify', 'GlobalAveragePooling',
    'LayerScale', 'LayerNorm', 'PermuteDims', 'get_kernel_spec',
    'xavier_uniform_init', 'kaiming_normal_init', 'neo_init'
]


def xavier_uniform_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        print(f"Xavier uniform init {m}")
        torch.nn.init.xavier_uniform(m.weight)


def kaiming_normal_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        print(f"Kaiming normal init {m}")
        torch.nn.init.kaiming_normal(m.weight)


def neo_init(m):
    """ NeoInit initialization method, see paper section 3.7 for the details """
    is_neocell = bool("NeoCell" in m.__class__.__name__)
    if is_neocell:
        with torch.no_grad():
            for wa in m.weight_a:
                c, h_out, h_in = wa.shape
                identity = skewed_identity(wa)
                wa.copy_(identity)
                wa.add_(torch.randn(c, h_out, h_in) / ((h_in*h_out)**0.5))
            for wb in m.weight_b:
                c, w_in, w_out = wb.shape
                identity = skewed_identity(wb)
                wb.copy_(identity)
                wb.add_(torch.randn(c, w_in, w_out) / ((w_in*w_out)**0.5))


def skewed_identity(data):
    _, n, m = data.shape
    w = torch.zeros_like(data)
    if m < n:
        step = round(n / m)
        for i in range(m):
            start = i*step
            end = start + step
            end = min(end, n)
            w[:, start:end, i] = 1.0
        w /= w.sum(axis=1, keepdims=True)
    elif m > n:
        step = round(m / n)
        for i in range(n):
            start = i*step
            end = start + step
            end = min(end, m)
            w[:, i, start:end] = 1.0
        w /= w.sum(axis=2, keepdims=True)
    elif n == m:
        for i in range(n):
            w[:, i, i] = 1.0
    return w


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        assert isinstance(patch_size, int)
        self.patch_size = patch_size

    def forward(self, x):
        n, c, h, w = x.shape
        assert h % self.patch_size == 0
        assert w % self.patch_size == 0
        h1 = h//self.patch_size
        w1 = w//self.patch_size
        x = x.reshape([n, c, h1, self.patch_size, w1, self.patch_size]) # NxCxH1xPxW1xP
        x = x.permute([0, 1, 3, 5, 2, 4]) # NxCxPxPxH1xW1
        return x.reshape([n, c*(self.patch_size**2), h1, w1]) # NxC*P*PxH1xW1


class PermuteDims(nn.Module):
    def __init__(self, permute_order):
        super().__init__()
        self.permute_order = permute_order

    def forward(self, x):
        return x.permute(self.permute_order)


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def __str__(self):
        return f"DropPath(p={self.drop_prob:.6f})"

    def __repr__(self):
        return self.__str__()


class ResidualCell(nn.Module):
    def __init__(self, content, drop_path):
        super().__init__()
        print(f"ResidualCell: drop_path = {drop_path:.3f}")
        self.content = content
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.content(x)
        return self.drop_path(y) + x


class LayerScale(nn.Module):
    def __init__(self, layer_scale_init_value, channels):
        super().__init__()
        print(f"LayerScale: layer_scale_init_value = {layer_scale_init_value}")
        self.gamma = None
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones((channels)),
                requires_grad=True
            )

    def forward(self, x):
        if self.gamma is not None:
            x = self.gamma * x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NeoBottleNeck(nn.Module):
    def __init__(
        self, channel_specs, c_outer,
        drop_path=0.0,
        inner_mult=4.0,
        layer_scale_init_value=0, linear_bias=False
    ):
        super().__init__()
        neo_channel_sum = sum([spec["channels"] for spec in channel_specs])
        assert neo_channel_sum == c_outer
        c_inner = int(c_outer * inner_mult)

        ops_list = [
            ('neocell', NeoCellCPP(channel_specs)),
            ('norm', nn.BatchNorm2d(c_outer)),
            ('chw2hwc', PermuteDims([0, 2, 3, 1])), # (N, C, H, W) -> (N, H, W, C)
            ('fc_1', nn.Linear(c_outer, c_inner, bias=linear_bias)),
            ('act', nn.GELU()),
            ('fc_2', nn.Linear(c_inner, c_outer, bias=linear_bias))
        ]
        if layer_scale_init_value > 0:
            ops_list.append(('layer_scale', LayerScale(layer_scale_init_value, c_outer)))
        ops_list.append(('hwc2chw', PermuteDims([0, 3, 1, 2]))) # (N, H, W, C) -> (N, C, H, W)

        self.ops = ResidualCell(nn.Sequential(OrderedDict(ops_list)), drop_path)

    def forward(self, x):
        profiler = get_nvtx_profiler()
        profiler.range_push("NeoBottleneck")

        out = self.ops(x)

        profiler.range_pop()
        return out


class NeoCellMatrices(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, x, channel_specs,
        a_full_height, a_full_width,
        b_full_height, b_full_width,
        *weights
    ):
        """
           Forward pass:
           y = (a @ x) @ b
        """
        weights_a = weights[:len(channel_specs)] # First half of the weights is a matrix A
        weights_b = weights[len(channel_specs):2 * len(channel_specs)] # Second half of the weights is a matrix B

        # prepare block diagonal weight matrices
        outputs = neocell_cpp.get_matrices_forward(
            weights_a, weights_b, channel_specs,
            a_full_height, a_full_width, b_full_height, b_full_width
        )

        matr_a_all, matr_b_all = outputs
        ctx.channel_specs = channel_specs
        ctx.a_full_height = a_full_height
        ctx.a_full_width = a_full_width
        ctx.b_full_height = b_full_height
        ctx.b_full_width = b_full_width

        # forward
        ax = matr_a_all.matmul(x)
        y = ax.matmul(matr_b_all)

        # save tensors for backward
        ctx.save_for_backward(x, matr_a_all, matr_b_all)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_y):
        """
        As a basic example of gradients of matrix multiplication you can refer to
        https://pytorch.org/docs/stable/notes/amp_examples.html#amp-custom-examples

        Backward pass:
        Transposing here (.T) - is swapping two last dimensions
        ∇b = (a @ x).T @ ∇y
        ∇(a @ x) = ∇y @ b.T
        ∇a = ∇(a @ x) @ x.T
        ∇x = a.T @ ∇(a @ x)

        Then neocell_cpp.get_matrices_backward computes gradients of the
        block-diagonalization operations
        """
        profiler = get_nvtx_profiler()
        profiler.range_push("NeoCellCPP_backward")

        # unpack saved tensors
        x, matr_a_all, matr_b_all = ctx.saved_tensors

        # backward
        ax = matr_a_all.matmul(x)
        grad_matr_b = ax.permute(0, 1, 3, 2).matmul(grad_y).sum(0)
        grad_ax = grad_y.matmul(matr_b_all.permute(0, 2, 1))
        grad_matr_a = grad_ax.matmul(x.permute(0, 1, 3, 2)).sum(0)
        grad_x = matr_a_all.permute(0, 2, 1).matmul(grad_ax)

        # accumulate gradients from tiles
        d_weight_a, d_weight_b = neocell_cpp.get_matrices_backward(
            grad_matr_a.contiguous(), grad_matr_b.contiguous(), ctx.channel_specs,
            ctx.a_full_height, ctx.a_full_width, ctx.b_full_height, ctx.b_full_width)

        profiler.range_pop()
        return tuple([grad_x] + [None]*5 + d_weight_a + d_weight_b)


def get_kernel_spec(spec_code, dims, shifts):
    """
        Generate kernel specification dictionary based on given spec_code, dims
        and shifts.

        Args:
            spec_code: string containing specification code. Should represent single
                integer (e.g. "4") or list of integers separated by "+" sign (e.g.
                "4+7+9"). In the second case dimensions (dims argument) is divided
                into number of kernels (e.g. to 3 in case of "4+7+9"), rounding down
                to nearest integer. If dims is not multiple of number of kernels dims
                for the last kernel is increased to fit the total number of dimaensions.
            dims: Number of input and output dimensions of NeoCell.
            shifts: 1 if spatial kernel shift is enabled, 0 otherwise.

        Returns:
            dict or List[dict]: contains kernel specifications in the following format
                {
                    "kernel": kernel_size,
                    "channels": channels_number,
                    "shift": enable_shift
                }
                where: `kernel_size` is size of NeoCell matrices, `channels_number` is
                the number of channels for this kernel_size and `enable_shift` indicates
                whether the spatial shift is enabled for this kernel_size

        Raises:
            ValueError: if spec_code does not represent single integer or list of
                integers separated by "+" sign
    """
    if "+" in spec_code:
        kernels = spec_code.split("+")
        try:
            kernels = [int(k) for k in kernels]
        except ValueError as e:
            print(
                f"get_kernel_spec: spec_code should be either integer, "
                f"or integers separated by '+' sign, got {spec_code}"
            )
            raise e
        n_kernels = len(kernels)
        sub_dim_size = dims // n_kernels
        sub_dims = [sub_dim_size]*n_kernels
        if dims % n_kernels != 0:
            last_sub_dim_size = dims - (n_kernels - 1)*sub_dim_size
            sub_dims[-1] = last_sub_dim_size
            print(f"get_kernel_spec warning: dims ({dims}) is not \
                    divisible by number of kernels ({n_kernels}).")
            print(f"get_kernel_spec warning: Last kernels number is increased \
                    from {sub_dim_size} to {last_sub_dim_size} to fit the data.")
        channels_spec = [
            {"kernel": kernels[i], "channels": sub_dims[i], "shift": shifts}
            for i in range(n_kernels)
        ]
    else:
        try:
            spec_code = int(spec_code)
            channels_spec = [{"kernel": spec_code, "channels": dims, "shift": shifts}]
        except ValueError as e:
            print(
                f"get_kernel_spec: spec_code should be either integer, "
                f"or integers separated by '+' sign, got {spec_code}"
            )
            raise e

    return channels_spec


class NeoCellCPP(nn.Module):
    """
        NeoCellCPP operator
        The same as NeoCellRepeatUnified, but with C++ implementation of matrix
        construction (approx. 2 times faster than Python version)
    """
    def __init__(self, channel_specs):
        """
            channel_specs - a list that contains dercriptions of the groups of
                NeoOperators in the form (heiht_in, heiht_out, width_in, width_out,
                channels, repeats, shift) or (kernel, channels, repeats, shift)
        """
        super().__init__()
        self.channel_specs = channel_specs
        self.sum_channels = sum([spec["channels"] for spec in self.channel_specs])

        weight_a = []
        weight_b = []

        self.a_full_width = 0
        self.a_full_height = 0
        self.b_full_width = 0
        self.b_full_height = 0

        for spec in self.channel_specs:
            h_in, h_out, w_in, w_out, channels = self.parse_spec(spec)
            weight_a.append(nn.Parameter(torch.randn(channels, h_out, h_in) / (h_out**0.5)))
            weight_b.append(nn.Parameter(torch.randn(channels, w_in, w_out) / (w_out**0.5)))

        self.weight_a = torch.nn.ParameterList(weight_a)
        self.weight_b = torch.nn.ParameterList(weight_b)

    def parse_spec(self, spec):
        # kernel size
        if "h_in" in spec:
            h_in  = spec["h_in"]
            h_out = spec["h_out"]
            w_in  = spec["w_in"]
            w_out = spec["w_out"]
        elif "kernel_a" in spec:
            h_in = h_out = spec["kernel_a"]
            w_in = w_out = spec["kernel_b"]
        else:
            h_in = h_out = w_in = w_out = spec["kernel"]
        # channels
        channels = spec["channels"]

        return h_in, h_out, w_in, w_out, channels

    def calculate_matrix_size(self, x):
        a_h, a_w, b_h, b_w = 0, 0, 0, 0
        for spec in self.channel_specs:
            h_in, h_out, w_in, w_out, _ = self.parse_spec(spec)
            # Matrix A
            if h_in == h_out:
                # square kernel does not change the size of the input
                a_h_spec = a_w_spec = x.size(2)
            else:
                a_w_spec = x.size(2)
                a_h_spec = math.floor((x.size(2) / float(h_in)) * h_out) # exact with rounding down

            # Matrix B
            if w_in == w_out:
                # square kernel does not change the size of the input
                b_h_spec = b_w_spec = x.size(3)
            else:
                b_h_spec = x.size(3)
                b_w_spec = math.floor((x.size(3) / float(w_in)) * w_out) # exact with rounding down

            a_h = max(a_h, a_h_spec)
            a_w = max(a_w, a_w_spec)
            b_h = max(b_h, b_h_spec)
            b_w = max(b_w, b_w_spec)

        if a_h == 0 or b_w == 0:
            raise RuntimeError(
                f"Given input size: {x.size()[1:]}. "
                f"Calculated output size: {(x.size(1), a_h, b_w)}. "
                "Output size is too small."
            )

        return a_h, a_w, b_h, b_w

    def forward(self, x):
        profiler = get_nvtx_profiler()
        profiler.range_push("NeoCellCPP")

        a_h, a_w, b_h, b_w = self.calculate_matrix_size(x)
        weights = []
        weights.extend(self.weight_a)
        weights.extend(self.weight_b)

        y = NeoCellMatrices.apply(
            x, self.channel_specs,
            a_h, a_w, b_h, b_w,
            *weights
        )

        profiler.range_pop()
        return y
