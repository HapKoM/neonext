import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_str, ActivationCountAnalysis

from ptvision.utils.parameters import parse_args
from ptvision.models.model_factory import get_model

from fvcore_hooks import _CUSTOM_SUPPORTED_OPS


def profile_fvcore(model, input_size=(3, 224, 224), batch_size=1, detailed=False, force_cpu=False):
    ''' Code from timm benchmark:
        https://github.com/rwightman/pytorch-image-models/blob/master/benchmark.py
    '''
    if force_cpu:
        model = model.to('cpu')

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=dtype)

    fca = FlopCountAnalysis(model, example_input).set_op_handle(**_CUSTOM_SUPPORTED_OPS)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size, params


if __name__ == "__main__":
    args = parse_args()
    args.drop_path = 0

    if not args.count_neocell:
        del _CUSTOM_SUPPORTED_OPS["prim::PythonOp.NeoCellMatrices"]

    torch.cuda.set_device(0)

    net: torch.nn.Module = get_model(args.model, args).cuda()
    # net.eval()

    macs, activations, params = profile_fvcore(
        net, (3, args.train_image_size, args.train_image_size),
        batch_size=args.per_batch_size,
        detailed=bool(args.verbose)
    )

    print('{:<30}  {:.4f} GMac'.format('Computational complexity: ', macs / 1e9))
    print('{:<30}  {:.2f} M'.format('Number of activations: ', activations / 1e6))
    print('{:<30}  {:.2f} M'.format('Number of parameters: ', params / 1e6))
