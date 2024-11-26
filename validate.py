import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

import sys
import os
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

from contextlib import suppress

import numpy as np

import torch

from ptvision.data.dataset_factory import get_datasets
from ptvision.models.model_factory import get_model
from ptvision.utils.parameters import parse_args
from ptvision.utils.distributed import get_local_rank
from ptvision.utils.model_ema import ModelEmaV2

from train_baseline import valid, prepare

if __name__ == "__main__":
    args = parse_args()
    args = prepare(args)

    eval_dataloader, _ = get_datasets(dataset_name=args.dataset_name, args=args, training=False)

    network = get_model(args.model, args=args)
    network.cuda()
    
    network_ema = None
    if args.ema:
        network_ema = ModelEmaV2(network, decay=args.ema_decay, device=None)

    # Calculate a number of parameters
    print(
        "Number of parameters: {:.2f} M".format(
            sum(p.numel() for p in network.parameters() if p.requires_grad) / float(10**6)
        )
    )

    # include fp32 training, amp fixed and dynamic loss scale
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress

    device_ids = [get_local_rank()]
    network = torch.nn.parallel.DistributedDataParallel(
        network, device_ids=device_ids, find_unused_parameters=False
    )

    if args.pretrain and os.path.exists(args.pretrain):
        ckpt_path = os.path.abspath(os.path.expanduser(args.pretrain))
        print(f"Loading checkpoint {ckpt_path}")
        pretrain_mod = torch.load(ckpt_path, map_location='cpu')
        m, u = network.load_state_dict(pretrain_mod, strict=True)
        if len(m) > 0:
            print(f"Missing keys: {m}")
        if len(u) > 0:
            print(f"Unexpected keys: {u}")
        if args.ema:
            assert ckpt_path.endswith(".pt")
            ckpt_path_ema = ckpt_path[:-3] + "_ema.pt"
            print(f"EMA enabled. Loading checkpoint {ckpt_path_ema}")
            pretrain_mod_ema = torch.load(ckpt_path_ema, map_location='cpu')
            m, u = network_ema.module.load_state_dict(pretrain_mod_ema, strict=True)
            if len(m) > 0:
                print(f"Missing keys: {m}")
            if len(u) > 0:
                print(f"Unexpected keys: {u}")
    else:
        print(f"Pretrained file {args.pretrain} not found")
        sys.exit(1)
    
    valid(network, eval_dataloader, amp_autocast)
    if args.ema:
        valid(network_ema.module, eval_dataloader, amp_autocast)
