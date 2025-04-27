import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.mixture import GaussianMixture

from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Scor
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef, MeanSquaredError

from src.data.dataset import DataAugment
from src.model.loss import NegEntropy
from src.utils.metrics_utils import MultilabelF1Max


def train_iteration(
    net_1: nn.Module,
    net_2: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_dataloader: torch.utils.data.DataLoader,
    unlabeled_dataloader: torch.utils.data.DataLoader,
    data_augment: DataAugment,
    better_model: int = 0,
    augmented_samples: int = 2,
    augment_scale: tuple = (0.0, 1.0),
    sharpening_temp: float = 20.0,
    alpha: float = 0.75,
    num_labels: int = 2,
    device: torch.device = torch.device("cuda"),
):
    net_1.train()
    net_2.eval()

    unlabeled_train_iter = iter(unlabeled_dataloader)

    epoch_loss = 0
    optimizer.zero_grad()
    for batch_idx, labeled_input_dict in tqdm.tqdm(enumerate(labeled_dataloader), desc="Training Iteration", leave=True):
        try:
            unlabeled_input_dict = next(unlabeled_train_iter)
            max_token = max(labeled_input_dict['embedding'].shape[1], unlabeled_input_dict['embedding'].shape[1])

            # pad to the same size
            if labeled_input_dict['embedding'].shape[1] < max_token:
                labeled_input_dict = pad(labeled_input_dict, max_token)
            elif unlabeled_input_dict['embedding'].shape[1] < max_token:
                unlabeled_input_dict = pad(unlabeled_input_dict, max_token)

            # ignore the last batch if it is not full
            if labeled_input_dict['embedding'].shape[0] != labeled_dataloader.batch_size:
                continue

        except:
            unlabeled_train_iter = iter(unlabeled_dataloader)
            unlabeled_input_dict = next(unlabeled_train_iter)
            max_token = max(labeled_input_dict['embedding'].shape[1], unlabeled_input_dict['embedding'].shape[1])

            # pad to the same size
            if labeled_input_dict['embedding'].shape[1] < max_token:
                labeled_input_dict = pad(labeled_input_dict, max_token)
            elif unlabeled_input_dict['embedding'].shape[1] < max_token:
                unlabeled_input_dict = pad(unlabeled_input_dict, max_token)

            # ignore the last batch if it is not full
            if labeled_input_dict['embedding'].shape[0] != labeled_dataloader.batch_size:
                continue

        labeled_input_dict = to_device(labeled_input_dict, device)
        unlabeled_input_dict = to_device(unlabeled_input_dict, device)
        batch_size = labeled_input_dict['embedding'].shape[0]
        w_clean = labeled_input_dict['pred']

        # make labels and guesses
        with torch.no_grad():
            # embedding augmentation
            noisy_embedding = []
            noisy_unlabeled_embedding = []
            for _ in range(augmented_samples):
                noisy_embedding.append(data_augment(labeled_input_dict['embedding'], mu=augment_scale[0], std=augment_scale[1]))
                noisy_unlabeled_embedding.append(data_augment(unlabeled_input_dict['embedding'], mu=augment_scale[0], std=augment_scale[1]))
            noisy_embedding = torch.stack(noisy_embedding, dim=1)  # (batch_size, n_samples, n_residues, c_res)
            noisy_unlabeled_embedding = torch.stack(noisy_unlabeled_embedding, dim=1)

            labeled_mask = torch.stack([labeled_input_dict['mask']] * augmented_samples, dim=1)
            unlabeled_mask = torch.stack([unlabeled_input_dict['mask']] * augmented_samples, dim=1)

            label = torch.nn.functional.one_hot(labeled_input_dict['label'], num_classes=num_labels).float()  # (batch_size, x)
            w_clean = w_clean.view(-1, 1)  # (batch_size, 1)

            # network prediction and aggregation
            if better_model == 0:
                pred = torch.mean(
                    torch.softmax(net_1(noisy_embedding, labeled_mask), dim=2),  # (batch_size, n_samples, x)
                    dim=1
                )  # (batch_size, x)
            elif better_model == 1:
                pred = torch.mean(
                    torch.softmax(net_1(noisy_embedding, labeled_mask), dim=2),  # (batch_size, n_samples, x)
                    dim=1
                )
            else:
                pred = torch.mean(
                    torch.softmax(net_2(noisy_embedding, labeled_mask), dim=2),  # (batch_size, n_samples, x)
                    dim=1
                )
            # label sharpening
            label = w_clean * label + (1 - w_clean) * pred  # (batch_size, x)
            label = label ** (1 / sharpening_temp)
            label = label / label.sum(dim=1, keepdim=True)
            label = label.detach()

            # network guessing and aggregation
            if better_model == 0:
                guess = torch.mean(
                    torch.cat(
                        [torch.softmax(net_1(noisy_unlabeled_embedding, unlabeled_mask), dim=2),
                         torch.softmax(net_2(noisy_unlabeled_embedding, unlabeled_mask), dim=2)], dim=1  # (batch_size, n_samples * 2, x)
                    ), dim=1  # (batch_size, x)
                )
            elif better_model == 1:
                guess = torch.mean(
                    torch.softmax(net_1(noisy_unlabeled_embedding, unlabeled_mask), dim=2), dim=1  # (batch_size, x)
                )
            else:
                guess = torch.mean(
                    torch.softmax(net_2(noisy_unlabeled_embedding, unlabeled_mask), dim=2), dim=1  # (batch_size, x)
                )
            # label sharpening
            # guess = guess ** (1 / sharpening_temp)
            guess = guess / guess.sum(dim=1, keepdim=True)
            guess = guess.detach()

        # mixmatch
        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        mixed_embedding = torch.cat(
            [noisy_embedding, noisy_unlabeled_embedding],
            dim=0
        ).view(-1, noisy_embedding.shape[2], noisy_embedding.shape[3])  # (batch_size * n_samples * 2, n_residues, c_res)
        mixed_mask = torch.cat([labeled_mask, unlabeled_mask], dim=0).view(-1, noisy_embedding.shape[2])
        mixed_label = torch.cat(
            [label] * augmented_samples + [guess] * augmented_samples,
            dim=0
        )  # (batch_size * n_samples * 2, x)

        augment_idx = torch.randperm(mixed_embedding.shape[0])
        mixed_embedding = l * mixed_embedding + (1 - l) * mixed_embedding[augment_idx]
        mixed_label = l * mixed_label + (1 - l) * mixed_label[augment_idx]
        label, guess = torch.split(mixed_label, batch_size * augmented_samples, dim=0)

        # forward pass
        pred = net_1(mixed_embedding, mixed_mask)  # (batch_size * n_samples * 2, x)
        pred_label, pred_guess = torch.split(pred, batch_size * augmented_samples, dim=0)

        # calculate loss
        loss_label = - 1 * torch.mean(
            torch.sum(label * torch.log_softmax(pred_label, dim=1), dim=1)  # (batch_size * n_samples)
        )  # (1)
        loss_guess = torch.mean((pred_guess - guess) ** 2)

        # regularization
        prior = torch.ones(num_labels) / num_labels
        prior = prior.to(device)
        pred_mean = torch.softmax(pred, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        # get sum loss
        loss = loss_label + loss_guess + penalty
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(labeled_dataloader)


def simple_train_iteration(
    net: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    labeled_dataloader: torch.utils.data.DataLoader,
    unlabeled_dataloader: torch.utils.data.DataLoader,
    num_labels: int = 2,
    task: str = "binary",  # options: "binary", "multi_class", "multi_label", "regression"
    device: torch.device = torch.device("cuda"),
):
    net.train()
    unlabeled_train_iter = iter(unlabeled_dataloader)
    epoch_loss = 0
    optimizer.zero_grad()

    for batch_idx, labeled_input_dict in tqdm.tqdm(enumerate(labeled_dataloader), desc="Training Iteration", leave=True):
        try:
            unlabeled_input_dict = next(unlabeled_train_iter)

            max_token = max(labeled_input_dict['embedding'].shape[1], unlabeled_input_dict['embedding'].shape[1])

            # pad to same length
            if labeled_input_dict['embedding'].shape[1] < max_token:
                labeled_input_dict = pad(labeled_input_dict, max_token)
            elif unlabeled_input_dict['embedding'].shape[1] < max_token:
                unlabeled_input_dict = pad(unlabeled_input_dict, max_token)

            if labeled_input_dict['embedding'].shape[0] != unlabeled_input_dict['embedding'].shape[0]:
                continue

        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_dataloader)
            continue

        labeled_input_dict = to_device(labeled_input_dict, device)
        unlabeled_input_dict = to_device(unlabeled_input_dict, device)
        batch_size = labeled_input_dict['embedding'].shape[0]
        w_clean = labeled_input_dict['pred'].view(-1, 1)

        # prepare embeddings and masks
        embedding_labeled = labeled_input_dict['embedding']
        embedding_unlabeled = unlabeled_input_dict['embedding']
        mask_labeled = labeled_input_dict['mask']
        mask_unlabeled = unlabeled_input_dict['mask']
        label = labeled_input_dict['label']

        # pseudo label generation
        with torch.no_grad():
            if task == "regression":
                label_pseudo = teacher(embedding_labeled, mask_labeled)
                label = w_clean * label.view(-1, 1) + (1 - w_clean) * label_pseudo
                guess = teacher(embedding_unlabeled, mask_unlabeled)
            else:
                output_labeled = teacher(embedding_labeled, mask_labeled)
                output_unlabeled = teacher(embedding_unlabeled, mask_unlabeled)
                if task == "binary" or task == "multi_class":
                    label_pseudo = torch.softmax(output_labeled, dim=1)
                    label = torch.nn.functional.one_hot(label, num_classes=num_labels).float()
                    label = w_clean * label + (1 - w_clean) * label_pseudo
                    label = label ** (1 / 0.5)
                    label = label / label.sum(dim=1, keepdim=True)
                    guess = torch.softmax(output_unlabeled, dim=1)
                elif task == "multi_label":
                    label_pseudo = torch.sigmoid(output_labeled)
                    label = w_clean * label + (1 - w_clean) * label_pseudo
                    label = label ** (1 / 0.5)
                    label = label / label.sum(dim=1, keepdim=True)
                    guess = torch.sigmoid(output_unlabeled)

        label = label.detach()
        guess = guess.detach()

        labeled_pred = net(embedding_labeled, mask_labeled)
        unlabeled_pred = net(embedding_unlabeled, mask_unlabeled)
        loss = loss_fn(labeled_pred, label) + (unlabeled_pred - guess) ** 2
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(labeled_dataloader)


def gmm_iteration(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    task: str = "binary",  # options: "binary", "multi_class", "multi_label", "regression"
    device: torch.device = torch.device("cuda"),
):
    net.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, input_dict in tqdm.tqdm(enumerate(loader), desc="GMM Iteration", leave=True):
            input_dict = to_device(input_dict, device)

            pred = net(input_dict['embedding'], input_dict['mask'])
            label = input_dict['label']
            if task == "regression":
                pred = pred.view(-1, 1)
                label = label.view(-1, 1)
            elif task == "multi_label":
                label = label.float()
            loss = loss_fn(pred, label)
            losses.append(loss)

    losses = torch.cat(losses, dim=0)
    losses = ((losses - losses.min()) / (losses.max() - losses.min())).view(-1, 1)
    losses = losses.cpu().numpy()

    # use GMM to guide next epoch training
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    return gmm.predict_proba(losses)[:, gmm.means_.argmin()], losses


def baseline_train_iteration(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    task: str = "binary",  # options: "binary", "multi_class", "multi_label", "regression"
    device: torch.device = torch.device("cuda"),
):
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    for batch_idx, input_dict in tqdm.tqdm(enumerate(dataloader), desc="Baseline Training Iteration", leave=True):
        input_dict = to_device(input_dict, device)

        pred = net(input_dict['embedding'], input_dict['mask'])
        label = input_dict['label']
        if task == "regression":
            pred = pred.view(-1, 1)
            label = label.view(-1, 1)
        elif task == "multi_label":
            label = label.float()
        loss = loss_fn(pred, label)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def baseline_val_iteration(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    task: str = "single_label",
    device: torch.device = torch.device("cuda"),
):
    net.eval()

    pred = []
    with torch.no_grad():
        for batch_idx, input_dict in tqdm.tqdm(enumerate(loader), desc="Validation Iteration", leave=True):
            input_dict = to_device(input_dict, device)
            pred.append(net(input_dict['embedding'], input_dict['mask']).detach().cpu())
    pred = torch.cat(pred, dim=0)
    if task == "binary":
        softmax_pred = torch.softmax(pred, dim=1)
        auc = BinaryAUROC(compute_on_step=False).to(device)
        acc = BinaryAccuracy(compute_on_step=False).to(device)
        recall = BinaryRecall(compute_on_step=False).to(device)
        precision = BinaryPrecision(compute_on_step=False).to(device)
        f1 = BinaryF1Score(compute_on_step=False).to(device)
        mcc = BinaryMatthewsCorrCoef(compute_on_step=False).to(device)
        metrics = {
            "auc": auc(pred, input_dict['label']),
            "acc": acc(softmax_pred, input_dict['label']),
            "recall": recall(softmax_pred, input_dict['label']),
            "precision": precision(softmax_pred, input_dict['label']),
            "f1": f1(softmax_pred, input_dict['label']),
            "mcc": mcc(softmax_pred, input_dict['label']),
        }
    elif task == "multi_class":
        softmax_pred = torch.softmax(pred, dim=1)
        auc = AUROC(num_classes=pred.shape[-1], compute_on_step=False).to(device)
        acc = Accuracy(num_classes=pred.shape[-1], compute_on_step=False).to(device)
        recall = Recall(num_classes=pred.shape[-1], compute_on_step=False).to(device)
        precision = Precision(num_classes=pred.shape[-1], compute_on_step=False).to(device)
        f1 = F1Scor(num_classes=pred.shape[-1], compute_on_step=False).to(device)
        mcc = MatthewsCorrCoef(num_classes=pred.shape[-1], compute_on_step=False).to(device)
        metrics = {
            "auc": auc(pred, input_dict['label']),
            "acc": acc(softmax_pred, input_dict['label']),
            "recall": recall(softmax_pred, input_dict['label']),
            "precision": precision(softmax_pred, input_dict['label']),
            "f1": f1(softmax_pred, input_dict['label']),
            "mcc": mcc(softmax_pred, input_dict['label']),
        }
    elif task == "multi_label":
        pred = torch.sigmoid(pred)
        max_f1 = MultilabelF1Max(num_labels=pred.shape[-1], compute_on_step=False).to(device)
        metrics = {
            "max_f1": max_f1(pred, input_dict['label']),
        }
    elif task == "regression":
        pred = pred.view(-1, 1)
        label = input_dict['label'].view(-1, 1)
        mse = MeanSquaredError(compute_on_step=False).to(device)
        spearman = SpearmanCorrCoef(compute_on_step=False).to(device)
        metrics = {
            "mse": mse(pred, input_dict['label']),
            "spearman": spearman(pred, input_dict['label']),
        }
    return metrics


def pad(
    input_feature_dict: dict,
    max_seq_len: int,
):
    for k, v in input_feature_dict.items():
        if len(v.shape) == 1:
            input_feature_dict[k] = v
        elif len(v.shape) == 2:
            input_feature_dict[k] = torch.cat(
                [v, torch.zeros(v.shape[0], max_seq_len - v.shape[1]).to(v.device)],
                dim=1)
        elif len(v.shape) == 3:
            input_feature_dict[k] = torch.cat(
                [v, torch.zeros(v.shape[0], max_seq_len - v.shape[1], v.shape[2]).to(v.device)],
                dim=1)
        else:
            raise Exception(f"shape {v.shape} not supported")
    return input_feature_dict


def to_device(obj, device):
    """Move tensor or dict of tensors to device"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    else:
        raise Exception(f"type {type(obj)} not supported")
    return obj
