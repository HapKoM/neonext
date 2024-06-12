import torch.distributed as dist
from torch.utils.data import Sampler
import math


class ValidateDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.rank = dist.get_rank()

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples
