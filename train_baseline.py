import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", category=ImportWarning)

import sys
import os
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

import time
import datetime
import builtins
from copy import deepcopy
from contextlib import suppress

import numpy as np

import torch
import torch.distributed as dist

from ptvision.data.dataset_factory import get_datasets
from ptvision.data.mixup import Mixup
from ptvision.models.model_factory import get_model
from ptvision.loss.loss_factory import get_loss
from ptvision.optim.optim_factory import get_optim
from ptvision.scheduler.lr_scheduler_factory import get_lr_scheduler
from ptvision.utils.parameters import parse_args
from ptvision.utils.seed import set_seed, set_deterministic
from ptvision.utils.logging import get_logger
from ptvision.utils.metrics import AverageMeter, accuracy_topk
from ptvision.utils.model_ema import ModelEmaV2
from ptvision.utils.distributed import (
    is_master, get_rank, get_local_rank, get_world_size
)

from cloud.mox_transfer import in_cloud, mox_copy


def valid(network, eval_dataloader, amp_autocast=suppress, prefix="", best_acc=None):
    t_start = time.time()
    network.eval()
    with torch.no_grad():
        correct = 0
        tot = 0
        rec_mse = None
        for imgs, target in eval_dataloader:
            imgs = imgs.cuda()
            target = target.cuda()
            with amp_autocast():
                out = network(imgs)
            if isinstance(out, (list, tuple)) and len(out) == 3:
                if rec_mse is None:
                    rec_mse = 0
                rec_out = out[2]
                rec_mse += (imgs - rec_out).square().mean(dim=[1, 2, 3]).sum()
            out = out[0] if isinstance(out, (list, tuple)) else out
            tot += imgs.size(0)
            acc = accuracy_topk(out, target)[0].item()
            correct += acc * imgs.size(0)

        total_images_tensor = torch.tensor(tot, dtype=torch.float).cuda()
        correct_tensor = torch.tensor(correct).cuda()
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_images_tensor, op=dist.ReduceOp.SUM)
        if rec_mse is not None:
            dist.all_reduce(rec_mse, op=dist.ReduceOp.SUM)
            rec_mse = rec_mse / total_images_tensor.item()
        valid_acc = correct_tensor.item() / total_images_tensor.item()
        if best_acc is not None and valid_acc > best_acc[0]:
            best_acc[0] = valid_acc
        log_str = prefix
        log_str += 'valid_acc={:.6f}'.format(valid_acc)
        log_str += ', correct={}'.format(int(correct_tensor.item()))
        log_str += ', img_num={}'.format(int(total_images_tensor.item()))
        if best_acc is not None:
            log_str += ', best_acc={:.6f}'.format(best_acc[0])
        if rec_mse is not None:
            log_str += ', rec. MSE={:.6f}'.format(rec_mse.item())
        log_str += ', time={:.2f}s'.format(time.time() - t_start)
        print(log_str)

    torch.cuda.empty_cache()
    return valid_acc


def prepare(base_args):
    args = deepcopy(base_args)

    torch.distributed.init_process_group(
        backend=args.backend,
        timeout=datetime.timedelta(hours=1.)
    )
    torch.cuda.set_device(get_local_rank())

    set_seed(args.seed)
    if args.deterministic:
        set_deterministic()

    # logger
    now_time = torch.from_numpy(np.array(float(time.time()))).cuda() / get_world_size()
    dist.all_reduce(now_time, op=dist.ReduceOp.SUM)
    now_time = datetime.datetime.fromtimestamp(
        now_time.cpu().numpy().tolist()
    ).strftime('%Y-%m-%d_%H-%M-%S')

    args.outputs_dir = os.path.join(args.outputs_dir, now_time)
    args.logger = get_logger(args.outputs_dir, get_rank())
    builtins.print = args.logger.info

    return args


def train(args):
    # dataloader
    # sampler update is handled in the PrefetchedWrapper
    train_dataloader, _ = get_datasets(dataset_name=args.dataset_name, args=args, training=True)
    eval_dataloader, _ = get_datasets(dataset_name=args.dataset_name, args=args, training=False)
    args.steps_per_epoch = len(train_dataloader)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smooth_factor, num_classes=args.num_classes)

    # network
    network = get_model(args.model, args=args)
    network.cuda()

    network_ema = None
    if args.ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        network_ema = ModelEmaV2(network, decay=args.ema_decay, device=None)

    if is_master():
        model_txt = os.path.join(args.outputs_dir, "model.txt")
        with open(model_txt, 'w') as fd:
            fd.write(str(network))
        if in_cloud():
            mox_copy(model_txt, os.path.join(args.train_url, "logs", "model.txt"))

    # Calculate a number of parameters
    print(
        "Number of parameters: {:.2f} M".format(
            sum(p.numel() for p in network.parameters() if p.requires_grad) / float(10**6)
        )
    )

    # loss
    criterion = get_loss(args.loss_name, args=args)
    criterion.cuda()

    # skip (schedule compression)
    if 'skip' not in args:
        args.skip = 0

    if args.skip > 0:
        assert args.skip < 1, "skip should be < 1.0"
        args.max_epoch = round((1 - args.skip) * args.max_epoch)
        if is_master():
            print(f"Train for {args.max_epoch} epochs (skip = {args.skip})")

    # optim
    optimizer = get_optim(args.optim_name, network=network, args=args)

    # loss_scale
    if args.amp and not args.is_dynamic_loss_scale:
        args.init_scale = args.static_loss_scale
        args.growth_interval = 10000000
        args.growth_factor = 1.00000001
        args.backoff_factor = 0.99999999

    # include fp32 training, amp fixed and dynamic loss scale
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    loss_scaler = torch.cuda.amp.GradScaler(
        init_scale=args.init_scale,
        growth_factor=args.growth_factor,
        backoff_factor=args.backoff_factor,
        growth_interval=args.growth_interval,
        enabled=args.amp
    )

    # lr scheduler
    lr_scheduler = get_lr_scheduler(args.lr_scheduler, optimizer=optimizer, args=args)

    args.logger.save_args(args)

    device_ids = [get_local_rank()]
    network = torch.nn.parallel.DistributedDataParallel(
        network, device_ids=device_ids, find_unused_parameters=False
    )

    # pretrain, TODO, in get_model or here
    # if os.path.exists(args.pretrained):
    #     pretrain_mod = torch.load(args.pretrained, map_location='cpu')
    #     network.load_state_dict(pretrain_mod, strict=True)

    print("Start training")
    best_acc = [-1, ]
    if args.ema:
        best_acc_ema = [-1, ]
    start_time = time.time()
    t_end = time.time()
    old_progress = 0
    current_step = 0
    loss_meter = AverageMeter('loss')
    train_acc_meter = AverageMeter('train_acc')

    if args.eval_first:
        valid(network, eval_dataloader, amp_autocast, 'before training: ', best_acc)
        if args.ema:
            valid(network_ema.module, eval_dataloader, amp_autocast, 'EMA before training: ', best_acc_ema)
        sys.exit(0)

    break_flag = False
    for epoch_idx in range(args.max_epoch):
        epoch_start = time.time()
        if break_flag:
            break

        network.train()
        train_dataiter = iter(train_dataloader)

        for _ in range(args.steps_per_epoch):
            if break_flag:
                break

            image, target = next(train_dataiter)
            image, target = image.cuda(), target.cuda()

            if mixup_fn is not None:
                image, target = mixup_fn(image, target)

            with amp_autocast():
                out = network(image)
                loss = criterion(out, target)

            train_acc = accuracy_topk(out[0] if isinstance(out, tuple) else out, target)[0]

            optimizer.zero_grad(set_to_none=True)
            loss_scaler.scale(loss).backward() # no work in not args.amp

            # gradient clipping
            if args.max_norm > 0:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), args.max_norm)

            loss_scaler.step(optimizer)
            loss_scaler.update()
            lr_scheduler.step()

            loss_meter.update(loss.item())
            train_acc_meter.update(train_acc.item())

            if network_ema is not None:
                network_ema.update(network)

            current_step += 1
            if current_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                time_used = time.time() - t_end
                fps = args.per_batch_size * (current_step - old_progress) * get_world_size() / time_used
                mem_reserved = "{:.1f}G".format(torch.cuda.memory_reserved() / (1024 ** 3))
                log_str = (
                    f'epoch_idx: {epoch_idx}, epoch: {current_step / args.steps_per_epoch:.2f}, iter: {current_step}'
                    f', {loss_meter}, {train_acc_meter}'
                    f', fps: {fps:.2f} imgs/sec, lr: {lr:.6f}'
                )
                if args.amp:
                    loss_scale = loss_scaler.get_scale()
                    log_str += ', loss_scale: {}'.format(loss_scale)
                    if loss_scale < 1e-2:
                        break_flag = True
                log_str += ", mem_reserved={}".format(mem_reserved)
                print(log_str)
                t_end = time.time()
                loss_meter.reset()
                train_acc_meter.reset()
                old_progress = current_step

        valid(
            network, eval_dataloader, amp_autocast,
            'epoch[{}], '.format(epoch_idx), best_acc
        )
        if network_ema is not None:
            valid(
                network_ema.module, eval_dataloader, amp_autocast,
                'EMA epoch[{}], '.format(epoch_idx), best_acc_ema
            )

        # copy logs to s3
        if in_cloud():
            args.logger.copy_log_to_s3(os.path.join(args.train_url, "logs"))

        if (
            args.save_ckpt and epoch_idx % args.save_ckpt_interval == 0 and
            is_master()
        ):
            ckpt_dir = os.path.join(args.outputs_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_fn = f'epoch_{epoch_idx}.pt'
            local_weights = os.path.join(ckpt_dir, ckpt_fn)
            torch.save(network.state_dict(), local_weights)
            if network_ema is not None:
                ckpt_fn_ema = f'epoch_{epoch_idx}_ema.pt'
                local_weights_ema = os.path.join(ckpt_dir, ckpt_fn_ema)
                torch.save(network_ema.module.state_dict(), local_weights_ema)

            if in_cloud():
                s3_ckpt_dir = os.path.join(args.train_url, "checkpoints")
                roma_weights_fp = os.path.join(s3_ckpt_dir, ckpt_fn)
                mox_copy(local_weights, roma_weights_fp)
                print("save weight success, roma_weights_fp: {}".format(roma_weights_fp))
                if network_ema is not None:
                    roma_weights_fp_ema = os.path.join(s3_ckpt_dir, ckpt_fn_ema)
                    mox_copy(local_weights_ema, roma_weights_fp_ema)
                    print("save EMA weight success, roma_weights_fp_ema: {}".format(roma_weights_fp_ema))
            else:
                print("save weight success, local_weights_fp: {}".format(local_weights))

        epoch_end = time.time()
        print("epoch time:{:.2f}s".format(epoch_end - epoch_start))

    valid(network, eval_dataloader, amp_autocast, 'finish training, ', best_acc)
    if network_ema is not None:
        valid(
            network_ema.module, eval_dataloader, amp_autocast,
            'EMA finish training, ', best_acc_ema
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('last_metric[{}]'.format(best_acc[0]))
    if network_ema is not None:
        print('last_metric_EMA[{}]'.format(best_acc_ema[0]))

    if in_cloud() and is_master():
        args.logger.copy_log_to_s3(os.path.join(args.train_url, "logs"))

    return break_flag


if __name__ == "__main__":
    args = parse_args()
    args = prepare(args)
    train(args)
