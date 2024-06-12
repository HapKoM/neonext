import numpy as np

import torch

import torchvision
from torchvision import transforms

from .sampler import ValidateDistributedSampler
from .autoaugment import ImageNetPolicy
from .randaugment import RandAugment
from .random_erasing import RandomErasing


class PrefetchedWrapper(object):
    def prefetched_loader(loader, num_classes, one_hot):
        mean = (
            torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
            .cuda()
            .view(1, 3, 1, 1)
        )

        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                next_input = next_input.float()
                if one_hot:
                    next_target = expand(num_classes, torch.float32, next_target)

                next_input = next_input.sub_(mean).div_(std)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, num_classes, one_hot):
        self.dataloader = dataloader
        self.epoch = 0
        self.one_hot = one_hot
        self.num_classes = num_classes

    def __iter__(self):
        if self.dataloader.sampler is not None and isinstance(
            self.dataloader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            self.dataloader.sampler.set_epoch(self.epoch)
        self.epoch += 1
        return PrefetchedWrapper.prefetched_loader(
            self.dataloader, self.num_classes, self.one_hot
        )

    def __len__(self):
        return len(self.dataloader)


def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=torch.contiguous_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets


def expand(num_classes, dtype, tensor):
    e = torch.zeros(
        tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
    )
    e = e.scatter(1, tensor.unsqueeze(1), 1.0)
    return e


def imagenet_dataset(data_dir, img_size=224, training=True, autoaug=False, randaug=False,
                     re_prob=0., re_mode='const', re_count=1, re_num_splits=0):
    if training:
        transform = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ]
        if autoaug:
            transform.append(ImageNetPolicy())
        if randaug:
            transform.append(RandAugment(num_ops=2, magnitude=9))
        # put totensor and normalize into PrefetchedWrapper
        if re_prob > 0.:
            print(f"RandomErasing({re_prob}, {re_mode}, {re_count}, {re_num_splits})")
            transform.append(transforms.ToTensor())
            transform.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))
            transform.append(transforms.ToPILImage())


        dataset = torchvision.datasets.ImageFolder(data_dir, transforms.Compose(transform))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        return dataset, sampler

    else:
        dataset = torchvision.datasets.ImageFolder(data_dir, transforms.Compose([
            transforms.Resize(int(round(img_size / 0.875))),
            transforms.CenterCrop(img_size),
        ]))
        # put totensor and normalize into PrefetchedWrapper
        sampler = ValidateDistributedSampler(dataset)
        return dataset, sampler


def imagenet_pytorch_dataloader(args, training=True):
    if training:
        dataset, sampler = imagenet_dataset(args.train_data_dir,
                                            img_size=args.train_image_size,
                                            training=True,
                                            autoaug=args.autoaugment,
                                            randaug=args.randaugment,
                                            re_prob=args.reprob,
                                            re_mode=args.remode,
                                            re_count=args.recount)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=args.per_batch_size,
                                                 sampler=sampler,
                                                 num_workers=args.train_num_workers,
                                                 pin_memory=True,
                                                 collate_fn=fast_collate,
                                                 drop_last=True)
    else:
        dataset, sampler = imagenet_dataset(args.eval_data_dir, args.eval_image_size, training=False)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=args.per_batch_size,
                                                 shuffle=False,
                                                 sampler=sampler,
                                                 num_workers=args.eval_num_workers,
                                                 pin_memory=True,
                                                 collate_fn=fast_collate,
                                                 drop_last=False)

    return PrefetchedWrapper(dataloader, args.num_classes, one_hot=False), sampler
