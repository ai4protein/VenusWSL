import datetime
import logging
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import DataAugment, ProteinDataset
from src.model.VenusWSL import PredictorPLM
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(args: DictConfig):
    logging_dir = os.path.join(args.logging_dir, f"TRAIN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if DIST_WRAPPER.rank == 0:
        # update logging directory with current time
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        os.makedirs(logging_dir)
        os.makedirs(os.path.join(logging_dir, "checkpoints"))  # for saving checkpoints

        # save current configuration in logging directory
        with open(f"{logging_dir}/config.yaml", "w") as f:
            OmegaConf.save(args, f)

    # check environment
    use_cuda = torch.cuda.device_count() > 0
    if use_cuda:
        device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
        devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
        logging.info(
            f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if DIST_WRAPPER.world_size > 1:
        logging.info(
            f"Using DDP with {DIST_WRAPPER.world_size} processes, rank: {DIST_WRAPPER.rank}"
        )
        timeout_seconds = int(os.environ.get("NCCL_TIMEOUT_SECOND", 600))
        dist.init_process_group(
            backend="nccl", timeout=datetime.timedelta(seconds=timeout_seconds)
        )
    # All ddp process got the same seed
    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_dataset,
        max_seq_len=args.data.max_seq_len,
    )

    # to get labeled and unlabeled dataloaders

    model_1 = PredictorPLM(
        plm_embedding_dim=1280,
        num_labels=2,
    ).to(device)
    model_2 = PredictorPLM(
        plm_embedding_dim=1280,
        num_labels=2,
    ).to(device)
    if DIST_WRAPPER.world_size > 1:
        logging.info("Using DDP")
        model_1 = DDP(
            model_1,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )
        model_2 = DDP(
            model_2,
            device_ids=[DIST_WRAPPER.local_rank],
            output_device=DIST_WRAPPER.local_rank,
            static_graph=True,
        )

    optimizer_1 = torch.optim.Adam(
        model_1.parameters(),
        lr=args.optimizer.lr,
    )
    optimizer_2 = torch.optim.Adam(
        model_2.parameters(),
        lr=args.optimizer.lr,
    )

    # to imply the main training loop


def train_iteration(
    net_1: nn.Module,
    net_2: nn.Module,
    optimizer: torch.optim.Optimizer,
    labeled_dataloader: torch.utils.data.DataLoader,
    unlabeled_dataloader: torch.utils.data.DataLoader,
    data_augment: DataAugment,
    augmented_samples: int = 2,
    sharpening_temp: float = 20.0,
    alpha: float = 0.75,
    num_labels: int = 2,
):
    net_1.train()
    net_2.eval()

    optimizer.zero_grad()
    for batch_idx, (labeled_input_dict, unlabeled_input_dict) in enumerate(zip(labeled_dataloader, unlabeled_dataloader)):
        device = net_1.device
        labeled_input_dict = to_device(labeled_dataloader, device)
        unlabeled_input_dict = to_device(unlabeled_dataloader, device)
        batch_size = labeled_input_dict['embedding'].shape[0]
        w_clean = torch.zeros_like(labeled_input_dict['label'])

        # make labels and guesses
        with torch.no_grad():
            # embedding augmentation
            noisy_embedding = []
            noisy_unlabeled_embedding = []
            for _ in range(augmented_samples):
                noisy_embedding.append(data_augment(labeled_input_dict['embedding']))
                noisy_unlabeled_embedding.append(data_augment(unlabeled_input_dict['embedding']))
            noisy_embedding = torch.stack(noisy_embedding, dim=1)  # (batch_size, n_samples, n_residues, c_res)
            noisy_unlabeled_embedding = torch.stack(noisy_unlabeled_embedding, dim=1)

            # network prediction and aggregation
            pred = torch.mean(
                torch.softmax(net_1(noisy_embedding), dim=2),  # (batch_size, n_samples, x)
                dim=1
            )  # (batch_size, x)
            # label sharpening
            label = w_clean * labeled_input_dict['label'] + (1 - w_clean) * pred
            label = label ** (1 / sharpening_temp)
            label = label.detach()

            # network guessing and aggregation
            guess = torch.mean(
                torch.cat(
                    [torch.softmax(net_1(noisy_unlabeled_embedding), dim=2),
                     torch.softmax(net_2(noisy_unlabeled_embedding), dim=2)], dim=1
                ), dim=1
            )
            # label sharpening
            guess = guess ** (1 / sharpening_temp)

        # mixmatch
        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        mixed_embedding = torch.cat([noisy_embedding, noisy_unlabeled_embedding], dim=0).view(batch_size * augmented_samples, -1, -1)
        mixed_label = torch.cat([label, guess], dim=0).view(batch_size * augmented_samples, -1)

        augment_idx = torch.randperm(mixed_embedding.shape[0])
        mixed_embedding = l * mixed_embedding + (1 - l) * mixed_embedding[augment_idx]
        mixed_label = l * mixed_label + (1 - l) * mixed_label[augment_idx]
        label, guess = torch.split(mixed_label, [batch_size, batch_size], dim=0)

        # forward pass
        pred = net_1(mixed_embedding)  # (batch_size * n_samples * 2, x)
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


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
