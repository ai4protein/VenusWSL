import torch
import torch.nn as nn
import numpy as np
import tqdm
from sklearn.mixture import GaussianMixture

from src.data.dataset import DataAugment
from src.model.loss import NegEntropy


def train_iteration(
    net_1: nn.Module,
    net_2: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_dataloader: torch.utils.data.DataLoader,
    unlabeled_dataloader: torch.utils.data.DataLoader,
    data_augment: DataAugment,
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
            max_token = max(labeled_input_dict['embedding'].shape[0], unlabeled_input_dict['embedding'].shape[0])

            # pad to the same size
            if labeled_input_dict['embedding'].shape[0] < max_token:
                labeled_input_dict = pad(labeled_input_dict, max_token)
            elif unlabeled_input_dict['embedding'].shape[0] < max_token:
                unlabeled_input_dict = pad(unlabeled_input_dict, max_token)

        except:
            unlabeled_train_iter = iter(unlabeled_dataloader)
            unlabeled_input_dict = next(unlabeled_train_iter)
            max_token = max(labeled_input_dict['embedding'].shape[0], unlabeled_input_dict['embedding'].shape[0])

            # pad to the same size
            if labeled_input_dict['embedding'].shape[0] < max_token:
                labeled_input_dict = pad(labeled_input_dict, max_token)
            elif unlabeled_input_dict['embedding'].shape[0] < max_token:
                unlabeled_input_dict = pad(unlabeled_input_dict, max_token)

        labeled_input_dict = to_device(labeled_dataloader, device)
        unlabeled_input_dict = to_device(unlabeled_dataloader, device)
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
            noisy_embedding = torch.cat(noisy_embedding, dim=0)  # (batch_size * n_samples, n_residues, c_res)
            noisy_unlabeled_embedding = torch.cat(noisy_unlabeled_embedding, dim=0)

            # network prediction and aggregation
            pred = torch.mean(
                torch.softmax(net_1(noisy_embedding, labeled_input_dict['mask']), dim=1),  # (batch_size * n_samples, x)
                dim=1
            )  # (batch_size, x)
            # label sharpening
            label = w_clean * labeled_input_dict['label'] + (1 - w_clean) * pred
            label = label ** (1 / sharpening_temp)
            label = label / label.sum(dim=1, keepdim=True)
            label = label.detach()

            # network guessing and aggregation
            guess = torch.mean(
                torch.cat(
                    [torch.softmax(net_1(noisy_unlabeled_embedding, unlabeled_input_dict['mask']), dim=1),
                     torch.softmax(net_2(noisy_unlabeled_embedding, unlabeled_input_dict['mask']), dim=1)], dim=0
                ), dim=1
            )
            # label sharpening
            guess = guess ** (1 / sharpening_temp)
            guess = guess / guess.sum(dim=1, keepdim=True)
            guess = guess.detach()

        # mixmatch
        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        mixed_embedding = torch.cat([noisy_embedding, noisy_unlabeled_embedding], dim=0)
        mixed_label = torch.cat([label, guess], dim=0)
        mixed_mask = torch.cat([labeled_input_dict['mask'], unlabeled_input_dict['mask']], dim=0)

        augment_idx = torch.randperm(mixed_embedding.shape[0])
        mixed_embedding = l * mixed_embedding + (1 - l) * mixed_embedding[augment_idx]
        mixed_label = l * mixed_label + (1 - l) * mixed_label[augment_idx]
        label, guess = torch.split(mixed_label, [batch_size, batch_size], dim=0)

        # forward pass
        pred = net_1(mixed_embedding, mixed_mask)  # (batch_size * n_samples * 2, x)
        pred_label, pred_guess = torch.split(pred, 2, dim=0)

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


def gmm_iteration(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device = torch.device("cuda"),
):
    net.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, input_dict in tqdm.tqdm(enumerate(loader), desc="GMM Iteration", leave=True):
            input_dict = to_device(input_dict, device)

            pred = net(input_dict['embedding'], input_dict['mask'])
            loss = loss_fn(pred, input_dict['label'])
            losses.append(loss)

    losses = torch.cat(losses, dim=0)
    losses = ((losses - losses.min()) / (losses.max() - losses.min())).view(-1, 1)
    losses = losses.cpu().numpy()

    # use GMM to guide next epoch training
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    return gmm.predict_proba(losses)[:, gmm.means_.argmin()]


def val_iteration(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda"),
):
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, input_dict in tqdm.tqdm(enumerate(loader), desc="Validation Iteration", leave=True):
            input_dict = to_device(input_dict, device)

            pred = net(input_dict['embedding'], input_dict['mask'])
            _, predicted = torch.max(pred, 1)

            total += input_dict['label'].size(0)
            correct += (predicted == input_dict['label']).sum().item()

    return 100. * correct / total


def baseline_train_iteration(
    net: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    penalty_fn: NegEntropy,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cuda"),
):
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    for batch_idx, input_dict in tqdm.tqdm(enumerate(dataloader), desc="Baseline Training Iteration", leave=True):
        input_dict = to_device(input_dict, device)

        pred = net(input_dict['embedding'], input_dict['mask'])
        loss = loss_fn(pred, input_dict['label']) + penalty_fn(pred)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def pad(
    input_feature_dict: dict,
    max_seq_len: int,
):
    for k, v in input_feature_dict.items():
        if len(v.shape) == 2:
            input_feature_dict[k] = torch.cat(
                [v, torch.zeros(v.shape[0], max_seq_len - v.shape[1]).to(v.device)],
                dim=1)
        else:
            input_feature_dict[k] = torch.cat(
                [v, torch.zeros(v.shape[0], max_seq_len - v.shape[1], v.shape[2]).to(v.device)],
                dim=1)
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
