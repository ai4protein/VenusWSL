import torch
from torch import nn


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


class LabeledDataLoss(object):
    def __call__(self, outputs, labels):
        return -1 * torch.mean(
            torch.sum(labels * torch.log_softmax(outputs, dim=1), dim=1)
        )


class UnlabeledDataLoss(object):
    def __call__(self, outputs, labels):
        return torch.mean((outputs - labels) ** 2)


class PriorPenalty(object):
    def __init__(self, num_labels: int):
        self.num_labels = num_labels
        self.prior = torch.ones(num_labels) / num_labels

    def __call__(self, outputs):
        self.prior = self.prior.to(outputs.device)
        pred_mean = torch.softmax(outputs, dim=1).mean(0)
        return torch.sum(self.prior * torch.log(self.prior / pred_mean))


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=1, reduction='mean', device="cuda"):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float32)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss