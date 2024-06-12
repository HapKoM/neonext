import torch.nn as nn
import torch
import torch.nn.functional as F


class CrossEntropySmooth(nn.Module):
    def __init__(self, num_classes=1000, smooth_factor=0., aux_factor=0.0, reduction='mean'):
        super(CrossEntropySmooth, self).__init__()
        self.smooth_factor = smooth_factor
        self.num_classes = num_classes
        self.on_value = 1 - smooth_factor
        self.off_value = smooth_factor / (self.num_classes - 1)
        self.reduction = reduction
        self.aux_factor = aux_factor

    def ce(self, logit, label):
        N = label.size(0)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
        if self.reduction == 'mean':
            loss = - torch.sum(log_prob * label) / N
        elif self.reduction == 'none':
            loss = - torch.sum(1.0 * log_prob * label, dim=1)
        return loss

    def forward(self, logits, label):
        N = label.size(0)
        smoothed_labels = torch.full(size=(N, self.num_classes), fill_value=self.off_value).to(label.device)
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(label, dim=1), value=self.on_value)

        if isinstance(logits, torch.Tensor):
            logit, aux_logit = logits, None
        else:
            logit, aux_logit = logits

        loss = self.ce(logit, smoothed_labels)
        if aux_logit is not None:
            loss = loss + self.aux_factor * self.ce(aux_logit, smoothed_labels)

        return loss


class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smooth_factor=0.):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smooth_factor
        self.smoothing = smooth_factor

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            target_v = torch.zeros_like(x)
            target_v.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1)
        else:
            target_v = target

        loss = torch.sum(-target_v * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
