import datetime
import logging
import os

import hydra
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import DataAugment, ProteinDataset, get_dataloader
from src.model.VenusWSL import PredictorPLM
from src.model.loss import NegEntropy, LabeledDataLoss, UnlabeledDataLoss, PriorPenalty
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def train(args: DictConfig):
    if args.training.baseline:
        flag = "BASELINE"
    else:
        flag = "TRAIN"
    logging_dir = os.path.join(args.logging_dir, f"{flag}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
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

    train_dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_training_set,
        max_seq_len=args.data.max_seq_len,
    )
    val_dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_validation_set,
        max_seq_len=args.data.max_seq_len,
    )

    labeled_dataloader = get_dataloader(
        dataset=train_dataset,
        batch_size=args.data.batch_size,
        shuffle=False,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        batch_size=args.data.batch_size,
        shuffle=False,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )

    model_1 = PredictorPLM(
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
    optimizer_1 = torch.optim.Adam(
        model_1.parameters(),
        lr=args.optimizer.lr,
    )

    baseline_loss = nn.CrossEntropyLoss()
    baseline_penalty = NegEntropy()
    dividemix_eval_loss = nn.CrossEntropyLoss(reduction='none')
    dividemix_penalty = PriorPenalty(num_labels=2)

    # to imply the main training loop
    if args.training.baseline:
        # initialize progress bar
        if DIST_WRAPPER.rank == 0:
            pbar = tqdm(range(args.epochs), desc="Training", leave=False, ncols=100)
            with open(f"{logging_dir}/loss.csv", "w") as f:
                f.write("epoch,loss,val_acc\n")
        for epoch in range(args.epochs):
            train_loss = baseline_train_iteration(
                model_1,
                optimizer_1,
                baseline_loss,
                baseline_penalty,
                labeled_dataloader,
            )
            val_acc = baseline_val_iteration(
                model_1,
                baseline_loss,
                val_dataloader,
            )
            if DIST_WRAPPER.rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f'{train_loss:.2f}', val_acc=f'{val_acc:.2f}')
                with open(f"{logging_dir}/loss.csv", "a") as f:
                    f.write(f"{epoch},{train_loss},{val_acc}\n")

                if epoch % args.training.save_interval == 0:
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_{epoch}.pt"))

    else:  # imply DivideMix
        model_2 = PredictorPLM(
            plm_embedding_dim=1280,
            num_labels=2,
        ).to(device)
        if DIST_WRAPPER.world_size > 1:
            model_2 = DDP(
                model_2,
                device_ids=[DIST_WRAPPER.local_rank],
                output_device=DIST_WRAPPER.local_rank,
                static_graph=True,
            )
        optimizer_2 = torch.optim.Adam(
            model_2.parameters(),
            lr=args.optimizer.lr,
        )
        data_augment = DataAugment(noise=True)

        gmm_dataloader = get_dataloader(
            dataset=train_dataset,
            batch_size=args.data.batch_size,
            shuffle=False,  # to make sure the calculated probability is consistent
            num_workers=args.data.num_workers,
            pin_memory=args.data.pin_memory,
        )

        # first warmup
        if DIST_WRAPPER.rank == 0:
            pbar = tqdm(range(args.training.warmup_epochs), desc="Warmup", leave=False, ncols=100)
            with open(f"{logging_dir}/loss_wsl.csv", "w") as f:
                f.write("epoch,loss,loss_2,val_acc,val_acc_2\n")
        for epoch in range(args.training.warmup_epochs):
            train_loss_1 = baseline_train_iteration(
                model_1,
                optimizer_1,
                baseline_loss,
                baseline_penalty,
                labeled_dataloader,
            )
            train_loss_2 = baseline_train_iteration(
                model_2,
                optimizer_2,
                baseline_loss,
                baseline_penalty,
                labeled_dataloader,
            )
            if DIST_WRAPPER.rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f'{train_loss_1:.2f}', loss_2=f'{train_loss_2:.2f}')
                with open(f"{logging_dir}/loss_wsl.csv", "a") as f:
                    f.write(f"{epoch},{train_loss_1},{train_loss_2},,\n")

                if epoch % args.training.save_interval == 0:
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_1_{epoch}.pt"))
                    torch.save(model_2.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_2_{epoch}.pt"))

        if DIST_WRAPPER.rank == 0:
            pbar = tqdm(range(args.epochs - args.training.warmup_epochs), desc="Training", leave=False, ncols=100)
        for epoch in range(args.training.warmup_epochs, args.epochs):
            prob_1 = gmm_iteration(
                model_1,
                gmm_dataloader,
                dividemix_eval_loss,
            )
            prob_2 = gmm_iteration(
                model_2,
                gmm_dataloader,
                dividemix_eval_loss,
            )
            prob_clean_1 = (prob_1 > args.training.p_threshold)
            prob_clean_2 = (prob_2 > args.training.p_threshold)
            train_loss_1 = train_iteration(
                model_1,
                model_2,
                optimizer_1,
                labeled_dataloader,
                unlabeled_dataloader,
                prob_2,
                data_augment,
                augmented_samples=args.training.augmented_samples,
                sharpening_temp=args.training.sharpening_temp,
                alpha=args.training.alpha,
                num_labels=2,
            )
            train_loss_2 = train_iteration(
                model_2,
                model_1,
                optimizer_2,
                labeled_dataloader,
                unlabeled_dataloader,
                prob_1,
                data_augment,
                augmented_samples=args.training.augmented_samples,
                sharpening_temp=args.training.sharpening_temp,
                alpha=args.training.alpha,
                num_labels=2,
            )
            val_acc_1 = val_iteration(
                model_1,
                val_dataloader,
            )
            val_acc_2 = val_iteration(
                model_2,
                val_dataloader,
            )
            if DIST_WRAPPER.rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f'{train_loss_1:.2f}', loss_2=f'{train_loss_2:.2f}',
                                 val_acc=f'{val_acc_1:.2f}', val_acc_2=f'{val_acc_2:.2f}')
                with open(f"{logging_dir}/loss.csv", "a") as f:
                    f.write(f"{epoch},{train_loss_1},{train_loss_2},{val_acc_1},{val_acc_2}\n")

                if epoch % args.training.save_interval == 0:
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_1_{epoch}.pt"))
                    torch.save(model_2.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_2_{epoch}.pt"))


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

    epoch_loss = 0
    optimizer.zero_grad()
    for batch_idx, (labeled_input_dict, unlabeled_input_dict) in enumerate(zip(labeled_dataloader, unlabeled_dataloader)):
        device = net_1.device
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
                noisy_embedding.append(data_augment(labeled_input_dict['embedding'], mu=0.0, std=1.0))
                noisy_unlabeled_embedding.append(data_augment(unlabeled_input_dict['embedding']))
            noisy_embedding = torch.cat(noisy_embedding, dim=0)  # (batch_size * n_samples, n_residues, c_res)
            noisy_unlabeled_embedding = torch.cat(noisy_unlabeled_embedding, dim=0)

            # network prediction and aggregation
            pred = torch.mean(
                torch.softmax(net_1(noisy_embedding), dim=1),  # (batch_size * n_samples, x)
                dim=1
            )  # (batch_size, x)
            # label sharpening
            label = w_clean * labeled_input_dict['label'] + (1 - w_clean) * pred
            label = label ** (1 / sharpening_temp)
            label = label.detach()

            # network guessing and aggregation
            guess = torch.mean(
                torch.cat(
                    [torch.softmax(net_1(noisy_unlabeled_embedding), dim=1),
                     torch.softmax(net_2(noisy_unlabeled_embedding), dim=1)], dim=0
                ), dim=1
            )
            # label sharpening
            guess = guess ** (1 / sharpening_temp)

        # mixmatch
        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        mixed_embedding = torch.cat([noisy_embedding, noisy_unlabeled_embedding], dim=0)
        mixed_label = torch.cat([label, guess], dim=0)

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
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return epoch_loss / len(labeled_dataloader)


def gmm_iteration(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
):
    net.eval()
    losses = []
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(loader):
            device = net.device
            input_dict = to_device(input_dict, device)

            pred = net(input_dict['embedding'])
            loss = loss_fn(pred, input_dict['label'])
            losses.append(loss.item())

            _, predicted = torch.max(pred, 1)

    losses = torch.cat(losses, dim=0)
    losses = ((losses - losses.min()) / (losses.max() - losses.min())).view(-1, 1)

    # use GMM to guide next epoch training
    gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    return gmm.predict_proba(losses)[:, gmm.means_.argmin()]


def val_iteration(
    net: nn.Module,
    loader: torch.utils.data.DataLoader,
):
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(loader):
            device = net.device
            input_dict = to_device(input_dict, device)

            pred = net(input_dict['embedding'])
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
):
    net.train()
    optimizer.zero_grad()

    epoch_loss = 0
    for batch_idx, input_dict in enumerate(dataloader):
        device = net.device
        input_dict = to_device(input_dict, device)

        pred = net(input_dict['embedding'])
        loss = loss_fn(pred, input_dict['label']) + penalty_fn(pred)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(dataloader)


def baseline_val_iteration(
    net: nn.Module,
    loss_fn: nn.Module,
    dataloader: torch.utils.data.Data
):
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, input_dict in enumerate(dataloader):
            device = net.device
            input_dict = to_device(input_dict, device)

            pred = net(input_dict['embedding'])
            _, predicted = torch.max(pred, 1)

            total += input_dict['label'].size(0)
            correct += (predicted == input_dict['label']).sum().item()
    return 100. * correct / total


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
