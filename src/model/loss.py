import torch


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