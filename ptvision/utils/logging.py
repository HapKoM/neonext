import os
import gc
import sys
import logging
import numpy as np

import torch


logger_name = 'pytorch_benchmark'


class LOGGER(logging.Logger):
    def __init__(self, logger_name, rank=0):
        super(LOGGER, self).__init__(logger_name)
        self.log_fn = None
        if rank % 8 == 0:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', "%Y-%m-%d %H:%M:%S")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir, rank=0):
        self.rank = rank
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = 'rank_{}.log'.format(rank)
        log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.log_fn = log_fn

    def copy_log_to_s3(self, train_url):
        try:
            import moxing as mox
            log_filename = os.path.basename(self.log_fn)
            roma_log_fp = os.path.join(train_url, log_filename)
            roma_log_dirname = os.path.dirname(roma_log_fp)
            if not mox.file.exists(roma_log_dirname):
                mox.file.make_dirs(roma_log_dirname)
            mox.file.copy(self.log_fn, roma_log_fp)
        except:
            pass

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            if len(args) > 0:
                if "%" in msg:
                    msg = str(msg) % args
                else:
                    msg = " ".join([str(msg)] + [str(x) for x in args])

            if "file" in kwargs.keys():
                kwargs.pop("file")

            if "end" in kwargs.keys():
                msg += kwargs["end"]
                kwargs.pop("end")

            if "flush" in kwargs.keys():
                kwargs.pop("flush")

            self._log(logging.INFO, msg, (), **kwargs)

    def save_args(self, args):
        self.info('Args:')
        if isinstance(args, (list, tuple)):
            for value in args:
                self.info('--> {}'.format(value))
        else:
            if isinstance(args, dict):
                args_dict = args
            else:
                args_dict = vars(args)
            for key in args_dict.keys():
                self.info('--> {}: {}'.format(key, args_dict[key]))
        self.info('')


def get_logger(path, rank=0):
    logger = LOGGER(logger_name, rank)
    logger.setup_logging_file(path, rank)
    return logger


def get_tensors_alive(regex=None, requires_grad=None, devices=None):
    """ Print all the tensors/parameters tracked by gc.
        Args:
            requires_grad: bool = None - by default include all the objects, False - show
                only non-trainable, True - show only trainable
            devices: Union[None, str, List[int]] = None - if None, print for all devices.
                If "cuda" print tensors on all GPUs. It's also possible to print tensors
                on the specific device(s). Example: devices=[-1, 0, 1] (-1 for "cpu";
                0,1 - GPU 0 and GPU 1)

        Returns:
            tensors - None if an error occured
            parameters - None if an error occured
    """
    n_gpus = torch.cuda.device_count()

    if devices is None:
        devices = ["cpu"] + [f"cuda:{i}" for i in range(n_gpus)]
    elif devices == "cpu":
        devices = ["cpu"]
    elif devices == "cuda":
        devices = [f"cuda:{i}" for i in range(n_gpus)]
    elif isinstance(devices, (tuple, list)):
        devices = sorted(np.unique(devices).tolist())
        if any((d < -1 or d >= n_gpus) for d in devices):
            return None, None
        devices = [
            "cpu" if d == -1 else f"cuda:{d}"
            for d in devices
        ]
    else:
        pass

    devices = [torch.device(d) for d in devices]
    tensors = {str(k): [] for k in devices}

    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) or isinstance(obj, torch.nn.Parameter):
                save_tensor = True
                if requires_grad is not None and obj.requires_grad != requires_grad:
                    save_tensor = False
                if obj.device not in devices:
                    save_tensor = False

                # TODO: check regex

                if save_tensor:
                    tensors[str(obj.device)].append(obj)
        except:
            pass

    return tensors


def print_tensors_alive(tensors: dict):
    for dev_key in sorted(tensors.keys()):
        for tensor in tensors[dev_key]:
            print(
                type(tensor),
                "shape:", tensor.size(),
                "Memory:", tensor.numel() * tensor.element_size() / 1024.0 / 1024, "MB"
            )
