import torch
from sklearn.metrics import roc_auc_score


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the accuracy of the predictions.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total if total > 0 else 0.0


def precision(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the precision of the predictions.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    true_positive = (pred * target).sum().item()
    predicted_positive = pred.sum().item()
    return true_positive / predicted_positive if predicted_positive > 0 else 0.0


def recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the recall of the predictions.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    true_positive = (pred * target).sum().item()
    actual_positive = target.sum().item()
    return true_positive / actual_positive if actual_positive > 0 else 0.0


def f1_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the F1 score of the predictions.
    """
    p = precision(pred, target)
    r = recall(pred, target)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def auc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the area under the curve (AUC) of the predictions.
    """
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    return roc_auc_score(target, pred)


def mcc(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate the Matthews correlation coefficient (MCC) of the predictions.
    """
    pred = pred.view(-1)
    target = target.view(-1)
    tp = (pred * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()

    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return numerator / denominator if denominator > 0 else 0.0