from .imagenet_pytorch import imagenet_pytorch_dataloader


def get_datasets(dataset_name, args, training):
    if dataset_name == "imagenet_pytorch":
        dataloader, sampler = imagenet_pytorch_dataloader(args, training=training)

    else:
        raise NotImplementedError

    # cache imgs to gpu memory for accelerate
    if not training and args.cache_eval:
        eval_list = []
        for imgs, target in dataloader:
            imgs = imgs.cuda()
            target = target.cuda()
            eval_list.append([imgs, target])
        dataloader = eval_list

    return dataloader, sampler
