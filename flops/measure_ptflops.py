import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

import torch

from ptflops import get_model_complexity_info

from ptvision.utils.parameters import parse_args
from ptvision.models.model_factory import get_model

from ptflops_hooks import CUSTOM_MODULES_HOOKS, NeoCellCPP


if __name__ == "__main__":
    args = parse_args()
    args.drop_path = 0

    torch.cuda.set_device(0)

    if not args.count_neocell:
        del CUSTOM_MODULES_HOOKS[NeoCellCPP]

    net: torch.nn.Module = get_model(args.model, args).cuda()
    # net.eval()

    macs, params = get_model_complexity_info(
        net, (3, args.train_image_size, args.train_image_size),
        custom_modules_hooks=CUSTOM_MODULES_HOOKS,
        as_strings=True, verbose=True,
        print_per_layer_stat=bool(args.verbose)
    )
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
