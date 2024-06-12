import os
import sys
import argparse

project_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, project_dir)

from cloud.mox_transfer import mox_copy
from cloud.node_sync import sync_through_remote_file, remove_remote_sync_files
from ptvision.utils.cfg_parser import parse_replace_roma


def exec_cmd(cmd, print_cmd=True):
    if print_cmd:
        print(cmd)
    os.system(cmd)


def setup_cloud_environment(args):
    if args.s3_cuda:
        local_path = args.s3_cuda.replace("s3://", "/cache/")
        mox_copy(args.s3_cuda, local_path)
        if local_path.endswith(('.tgz', '.tar')):
            exec_cmd(f"tar xf {local_path} --directory /cache")

    # setup environment
    exec_cmd('lscpu')
    exec_cmd("sudo chown -R work:work /home/work")
    exec_cmd('pip install -U pip')
    exec_cmd('cd {}; pip install -r requirements.txt'.format(project_dir))

    # replace Pillow with faster Pillow-SIMD compiled
    # with the support of AVX2 instructions
    if args.pillow_simd:
        exec_cmd('pip uninstall -y pillow pillow-simd')
        exec_cmd('CC="cc -mavx2" pip install --no-cache-dir pillow-simd')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    # --------------------distributed parameter---------------------
    parser.add_argument('--backend', type=str, default="nccl", help='use for current backend to support distributed')
    parser.add_argument('--init_method', type=str, default="tcp://127.0.0.1:50717", help='init method to support distributed')
    parser.add_argument('--rank', type=int, default=0, help='rank')
    parser.add_argument('--world_size', type=int, default=1, help='current process number to support distributed')

    # -----------------------cloud parameter-----------------------
    parser.add_argument('--train_url', type=str, default="", help='train url')

    # -----------------------train parameter-----------------------
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--pillow_simd', type=int, default=1, help='install Pillow-SIMD')
    parser.add_argument('--s3_cuda', type=str, default='', help='copy cuda libraries')
    parser.add_argument('--script', type=str, default='train_baseline.py',
                        help='training script to run')

    args, unknown = parser.parse_known_args()

    if args.train_url == '':
        exit(0)

    setup_cloud_environment(args)

    # prepare for possible remote multi-node sync
    remove_remote_sync_files(args.train_url, args.rank)

    # copy data from s3 to cache
    parse_replace_roma(
        os.path.join(project_dir, args.config),
        copy_to_cache=True
    )

    # multi node
    if args.world_size > 1:
        rank = args.rank
        world_size = args.world_size
        sync_through_remote_file(args.train_url, rank, world_size)

    import torch

    ngpus_per_node = torch.cuda.device_count()
    host, port = args.init_method.split(":")[1:]
    host = host.lstrip('/')

    cmd_launcher = (
        f'cd {project_dir}; '
        f'torchrun'
        f' --node_rank={args.rank}'
        f' --nnodes={args.world_size}'
        f' --nproc_per_node={ngpus_per_node}'
        f' --rdzv_backend=static'
        f' --rdzv_endpoint={host}:{port}'
    )

    cmd = (
        f' {args.script}'
        f' --backend={args.backend}'
        f' --train_url={args.train_url}'
        f' --config={args.config}'
    )
    for it in unknown:
        it = it.replace('--', '')
        val_list = it.split('=')
        if len(val_list) == 1:
            key, val = val_list[0], ''
        elif len(val_list) == 2:
            key, val = val_list
        else:
            raise ValueError
        cmd += ' --{}={}'.format(key, val)

    exec_cmd(cmd_launcher + cmd)
