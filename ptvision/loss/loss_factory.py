from torch import nn

from .cross_entropy import CrossEntropySmooth, NLLMultiLabelSmooth, SoftTargetCrossEntropy


def get_loss(loss_name, args):
    if loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()

    elif loss_name == 'ce_smooth':
        criterion = CrossEntropySmooth(num_classes=args.num_classes,
                                       smooth_factor=args.label_smooth_factor)
    elif loss_name == 'ce_smooth_mixup':
        criterion = NLLMultiLabelSmooth(smooth_factor=args.label_smooth_factor)
    elif loss_name == 'soft_target_ce':
        criterion = SoftTargetCrossEntropy()
    else:
        raise NotImplementedError

    return criterion
