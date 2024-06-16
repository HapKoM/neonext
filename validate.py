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

from train_baseline import valid, prepare

if __name__ == "__main__":
    args = parse_args()
    args = prepare(args)

    eval_dataloader, _ = get_datasets(dataset_name=args.dataset_name, args=args, training=False)

    network = get_model(args.model, args=args)
    network.cuda()

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

    if os.path.exists(args.pretrain):
        print(f"Loading pretrained file {args.pretrain}")
        pretrain_mod = torch.load(args.pretrain, map_location='cpu')
        network.load_state_dict(pretrain_mod, strict=True)
    else:
        print(f"Pretrained file {args.pretrain} not found")
        sys.exit(1)
    
    valid(network, eval_dataloader, amp_autocast)
