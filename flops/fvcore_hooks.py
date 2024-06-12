from numbers import Number
from typing import Any, List

from fvcore.nn.jit_handles import get_shape


def neocell_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the NeoCell operation.

    inputs: [x, A1, A2, ..., An, B1, B2, ..., Bn], where n - number of kernels in the NeoCell
    outputs: [y, ]
    """
    flop = 0
    bs, c, h_in, w_in = get_shape(inputs[0])
    bs1, c1, h_out, w_out = get_shape(outputs[0])

    assert bs == bs1 and c == c1, ((bs, bs1), (c, c1))
    # matrix A - forward
    flop += c * h_out * h_in * w_in
    # matrix B - forward
    flop += c * h_out * w_in * w_out

    flop *= bs
    return flop


_CUSTOM_SUPPORTED_OPS = {
    "prim::PythonOp.NeoCellMatrices": neocell_flop_jit,
}
