import os
import time
from pathlib import Path

from .mox_transfer import mox_copy


def _check_dir(dist_dir):
    copy_flag = True
    if os.path.exists(dist_dir):
        copy_flag = False
    if not os.path.exists(os.path.dirname(dist_dir)):
        os.makedirs(os.path.dirname(dist_dir))
    return copy_flag


def copy_data_to_cache(src_dir='', dist_dir=''):
    start_t = time.time()
    copy_flag = _check_dir(dist_dir)
    if copy_flag:
        print('copy from {}'.format(src_dir))
        tar_files = []
        t0 = time.time()
        if ".mindrecord" in src_dir:
            src_dir = os.path.split(src_dir)[0]
            dist_dir = os.path.split(dist_dir)[0]

        mox_copy(src_dir, dist_dir)
        if dist_dir.endswith('tar') or dist_dir.endswith('tar.gz'):
            tar_files.append(dist_dir)

        t1 = time.time()
        print('copy datasets, time used={:.2f}s'.format(t1 - t0))
        tar_list = list(Path(dist_dir).glob('**/*.tar'))
        tar_files.extend(tar_list)
        tar_list = list(Path(dist_dir).glob('**/*.tar.gz'))
        tar_files.extend(tar_list)
        # tar xvf tar file
        print('tar_files:{}'.format(tar_files))
        for tar_file in tar_files:
            tar_dir = os.path.dirname(tar_file)
            print('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
            os.system('cd {}; tar -xvf {} > /dev/null 2>&1'.format(tar_dir, tar_file))
            t2 = time.time()
            print('uncompress, time used={:.2f}s'.format(t2 - t1))
            os.system('cd {}; rm -rf {}'.format(tar_dir, tar_file))

        print('--------------------------------------------')
        print('copy data completed!')
    else:
        print("Since data already exists, copying is not required")
    end_t = time.time()
    print('copy cost time {:.2f} sec'.format(end_t-start_t))
