from torch import nn


def _split_params(network, args):
    # TODO: do not use specific model sources if possible
    # put imports inside to avoid building extensions in DataLoader workers
    from ptvision.models.neonext.neonext_utils import LayerScale, LayerNorm

    group_dict = {'wd':[], 'no_wd':[]}

    if isinstance(network, nn.Module):
        network = [network, ]

    for sub_network in network:
        for name, m in sub_network.named_modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.Parameter)):
                group_dict['wd'].append(m.weight)
                if m.bias is not None:
                    group_dict['no_wd'].append(m.bias)

            elif isinstance(
                m, (
                    nn.BatchNorm2d, nn.BatchNorm1d,
                    nn.LayerNorm, LayerNorm
                )
            ):
                if m.weight is not None:
                    group_dict['no_wd'].append(m.weight)
                if m.bias is not None:
                    group_dict['no_wd'].append(m.bias)

            elif isinstance(m, nn.PReLU):
                group_dict['no_wd'].append(m.weight)
            elif hasattr(m, 'weight_a'): # NeoCell
                for w in m.weight_a:
                    group_dict['wd'].append(w)
                for w in m.weight_b:
                    group_dict['wd'].append(w)
                for w in getattr(m, 'weight_a_g', []):
                    group_dict['no_wd'].append(w)
                for w in getattr(m, 'weight_b_g', []):
                    group_dict['no_wd'].append(w)
            elif isinstance(m, LayerScale):
                if m.gamma is not None:
                    group_dict['no_wd'].append(m.gamma)
            else:
                pass

    p_dict = {}
    for sub_network in network:
        for name, m in sub_network.named_parameters():
            p_dict[m] = name

    for vs in group_dict.values():
        for v in vs:
            if v in p_dict.keys():
                p_dict[v] = 0

    tot_parameters = sum([len(list(sub_network.parameters())) for sub_network in network])
    if tot_parameters != sum([len(x) for x in group_dict.values()]):
        p_dict = {}
        for sub_network in network:
            for name, m in sub_network.named_parameters():
                p_dict[m] = name
        for vs in group_dict.values():
            for v in vs:
                if v in p_dict.keys():
                    p_dict[v] = None
        print(
            'parameter not include: {}'.format(
                [x for x in p_dict.values() if x is not None]
            )
        )
        assert False

    groups = [
        dict(params=group_dict['wd']),
        dict(params=group_dict['no_wd'], weight_decay=.0),
    ]
    return groups
