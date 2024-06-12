import argparse

from cloud.mox_transfer import in_cloud
from .cfg_parser import merge_args


def parse_args():
    parser = argparse.ArgumentParser('pytorch imagenet training')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='imagenet_pytorch', help='dataset_name')
    parser.add_argument('--per_batch_size', default=256, type=int, help='batch size for per gpu')
    parser.add_argument('--train_num_workers', default=8, type=int, help='train_num_workers')
    parser.add_argument('--eval_num_workers', default=8, type=int, help='eval_num_workers')
    # dataset: path in cloud and local
    parser.add_argument('--train_data_dir', type=str, default='/ssd/ssd0/datasets/imagenet/train', help='data dir')
    parser.add_argument('--eval_data_dir', type=str, default='/ssd/ssd0/datasets/imagenet/val', help='data dir')
    parser.add_argument('--local_train_data_dir', type=str, default='/ssd/ssd0/datasets/imagenet/train', help='data dir')
    parser.add_argument('--local_eval_data_dir', type=str, default='/ssd/ssd0/datasets/imagenet/val', help='data dir')
    # dataset: imagenet
    parser.add_argument('--train_image_size', type=int, default=224, help='image size of the dataset')
    parser.add_argument('--eval_image_size', type=int, default=224, help='evaluate image size of the dataset')
    parser.add_argument('--autoaugment', type=int, default=0, help='autoaugment')
    parser.add_argument('--randaugment', type=int, default=0, help='randaugment')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # dataset: cache val dataset in HBM
    parser.add_argument('--cache_eval', default=0, type=int, help='eval_accelerate')
    parser.add_argument('--eval_first', default=0, type=int, help='eval before training')

    # network related
    parser.add_argument('--model', default='resnet50', type=str, help='backbone')
    parser.add_argument('--pretrain', default='', type=str, help='model_path, local pretrain model to load')
    # network: classification
    parser.add_argument('--num_classes', default=1000, type=int, help='num_classes')

    # loss related
    parser.add_argument('--loss_name', default='ce_smooth', type=str, help='loss_name')
    parser.add_argument('--label_smooth_factor', default=0.1, type=float, help='label_smooth_factor')

    # lr related
    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='lr-scheduler, option type: step, cosine')
    parser.add_argument('--lr_max', default=0.4, type=float, help='lr_max')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='weight decay')
    parser.add_argument('--warmup_init_lr', default=0.1, type=float, help='warmup lr')
    parser.add_argument('--warmup_epochs', default=5, type=float, help='warmup epoch')
    # lr: step
    parser.add_argument('--lr_epochs', type=str, default='30,60,80', help='epoch of lr changing')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='decrease lr by a factor of exponential lr_scheduler')
    # lr: cosine
    parser.add_argument('--eta_min', type=float, default=0., help='eta_min in cosine scheduler')

    # optimizer
    parser.add_argument('--optim_name', type=str, default='sgd', help='optim_name')
    parser.add_argument('--max_epoch', type=int, default=90, help='max_epoch')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--bn_weight_decay', type=int, default=0, help='if bn_weight_decay')
    # optimizer: rmsprop
    parser.add_argument('--rmsprop_alpha', type=float, default=0.9, help='rmsprop_alpha')
    parser.add_argument('--rmsprop_eps', type=float, default=0.01, help='rmsprop_eps')

    # amp and loss_scale
    parser.add_argument('--amp', default=0, type=int, help='amp for training model')
    parser.add_argument('--max_norm', default=0, type=float, help='gradient clipping threshold')
    # amp: on
    parser.add_argument('--is_dynamic_loss_scale', type=int, default=1, help='dynamic loss scale')
    # fixed loss scale
    parser.add_argument('--static_loss_scale', type=int, default=1024, help='static loss scale')
    # dynamic loss scale
    parser.add_argument('--init_scale', type=float, default=65536, help='init_scale')
    parser.add_argument('--growth_factor', type=float, default=2, help='growth_factor')
    parser.add_argument('--backoff_factor', type=float, default=0.5, help='backoff_factor')
    parser.add_argument('--growth_interval', type=int, default=2000, help='growth_interval')

    # logging and ckpt related
    parser.add_argument('--outputs_dir', type=str, default="outputs", help='output_dir')
    parser.add_argument('--log_interval', type=int, default=100, help='logging interval')
    parser.add_argument('--save_ckpt', type=int, default=0, help='save_ckpt')
    parser.add_argument('--save_ckpt_interval', type=int, default=1, help='save_interval')

    # distributed related
    parser.add_argument('--backend', type=str, default="nccl", help='use for current backend to support distributed')

    # other
    parser.add_argument('--config', type=str, required=True, help='yml config')
    parser.add_argument('--train_url', type=str, default="", help='train url')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--deterministic', type=int, default=0, help='deterministic')

    parser.add_argument('--rec_loss_factor', type=float, default=1.0, help='Coefficient for reconstruction loss')

    parser.add_argument('--force_fp16', type=int, default=0, help='create fp16 matrices in NeoCell')
    parser.add_argument('--drop_path', type=float, default=0.0, help='DropPath rate')

    parser.add_argument('--ema', type=int, default=0, help='Use EMA?')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay (if used)')

    # FLOPs
    parser.add_argument('--count_neocell', type=int, default=1, help='count NeoCell, 0 - consider zero-op')
    parser.add_argument('--verbose', type=int, default=0, help='print detailed log if available')

    parser.add_argument('--conv_init_type', type=str, default='', help='Conv2D init method (None|xavier_uniform|kaiming_normal)')
    parser.add_argument('--shifts', type=str, default="1,1,1,0", help='Shifts flags in stages')
    parser.add_argument('--layer_scale_init_value', type=float, default=0.0, help='Layer scale initial value (if 0.0, layer scale is not used)')
    parser.add_argument('--linear_bias', type=int, default=0, help='Use bias in the linear layers?')
    parser.add_argument('--kernel_spec', type=str, default="default", help='NeoCell kernel specification')

    args, _ = parser.parse_known_args()
    args = merge_args(args, args.config)

    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.shifts = [int(a) for a in args.shifts.split(",")]

    if in_cloud():
        args.outputs_dir = "outputs/"
    else:
        args.train_data_dir = args.local_train_data_dir
        args.eval_data_dir = args.local_eval_data_dir

    return args
