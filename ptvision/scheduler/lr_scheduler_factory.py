from .scheduler import WarmupCosine, WarmupStep, WarmupPoly


def get_lr_scheduler(lr_scheduler_name, optimizer, args):
    if lr_scheduler_name == 'step':
        lr_scheduler = WarmupStep(optimizer,
                                  args.max_epoch,
                                  args.steps_per_epoch,
                                  args.warmup_epochs,
                                  args.warmup_init_lr,
                                  args.lr_epochs,
                                  args.lr_gamma)

    elif lr_scheduler_name == 'cosine':
        lr_scheduler = WarmupCosine(optimizer,
                                    args.max_epoch,
                                    args.steps_per_epoch,
                                    args.warmup_epochs,
                                    args.warmup_init_lr,
                                    args.eta_min)

    elif lr_scheduler_name == 'poly':
        lr_scheduler = WarmupPoly(optimizer,
                                  args.max_epoch,
                                  args.steps_per_epoch,
                                  args.warmup_epochs,
                                  args.warmup_init_lr,
                                  poly_pow=2)

    else:
        raise NotImplementedError

    return lr_scheduler
