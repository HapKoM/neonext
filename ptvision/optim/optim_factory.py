import torch

from .utils import _split_params


def get_optim(optim_name, network, args, params_group=None):
    if params_group is None:
        if args.bn_weight_decay:
            print('batchnorm will use weight decay!!!')
            groups = network.parameters()
        else:
            print('batchnorm will not use weight decay!!!')
            groups = _split_params(network, args)
    else:
        groups = params_group

    lr = args.lr_max
    weight_decay = args.weight_decay
    optim_name = optim_name.lower()

    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(groups,
                                    lr=lr,
                                    momentum=args.momentum,
                                    weight_decay=weight_decay)

    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(groups,
                                      lr=lr,
                                      weight_decay=weight_decay)

    elif optim_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(groups,
                                        lr=lr,
                                        alpha=args.rmsprop_alpha,
                                        eps=args.rmsprop_eps,
                                        momentum=args.momentum,
                                        weight_decay=weight_decay)

    return optimizer
