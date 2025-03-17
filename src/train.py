import datetime
import logging
import os
import warnings

import hydra
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
import rootutils

from src.data.dataset import DataAugment, ProteinDataset, BatchTensorConverter
from src.model.VenusWSL import PredictorPLM
from src.model.loss import NegEntropy, LabeledDataLoss, UnlabeledDataLoss, PriorPenalty
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.training_utils import (baseline_train_iteration,
                                      gmm_iteration,
                                      train_iteration,
                                      val_iteration)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
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

    labeled_dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_training_set,
        max_seq_len=args.data.max_seq_len,
        batch_size=args.batch_size,
        collate_fn=BatchTensorConverter(),
        shuffle=args.data.shuffle,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )
    val_dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_validation_set,
        max_seq_len=args.data.max_seq_len,
        batch_size=args.batch_size,
        collate_fn=BatchTensorConverter(),
        shuffle=False,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )

    labeled_dataloader = labeled_dataset.get_dataloader()
    val_dataloader = val_dataset.get_dataloader()

    model_1 = PredictorPLM(
        plm_embed_dim=args.model.embedding_dim,
        attn_dim=args.model.attn_dim,
        num_labels=args.model.label_dim,
    ).to(device)

    if DIST_WRAPPER.rank == 0:
        logging.info(model_1)
        logging.info(f"Number of parameters: {sum(p.numel() for p in model_1.parameters())}")

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
        val_acc_best = 0.
        for epoch in range(args.epochs):
            torch.cuda.empty_cache()
            train_loss = baseline_train_iteration(
                model_1,
                optimizer_1,
                baseline_loss,
                baseline_penalty,
                labeled_dataloader,
                device=device,
            )
            val_acc = val_iteration(
                model_1,
                val_dataloader,
                device=device,
            )
            if DIST_WRAPPER.rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f'{train_loss:.2f}', val_acc=f'{val_acc:.2f}')
                with open(f"{logging_dir}/loss.csv", "a") as f:
                    f.write(f"{epoch},{train_loss},{val_acc}\n")

                if epoch % args.training.save_interval == 0:
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_{epoch}.pt"))
                if val_acc > val_acc_best:
                    val_acc_best = val_acc
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", "best.pt"))

    else:  # imply DivideMix
        model_2 = PredictorPLM(
            plm_embed_dim=args.model.embedding_dim,
            attn_dim=args.model.attn_dim,
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

        unlabeled_dataset = labeled_dataset.clone()
        unlabeled_dataset_2 = labeled_dataset.clone()
        labeled_dataset_2 = labeled_dataset.clone()
        labeled_dataloader_2 = labeled_dataset_2.get_dataloader()

        gmm_dataset = ProteinDataset(
            path_to_dataset=args.data.path_to_training_set,
            max_seq_len=args.data.max_seq_len,
            batch_size=args.batch_size,
            collate_fn=BatchTensorConverter(),
            shuffle=False,
            num_workers=args.data.num_workers,
            pin_memory=args.data.pin_memory,
        )
        gmm_dataloader = gmm_dataset.get_dataloader()

        # first warmup
        if DIST_WRAPPER.rank == 0:
            pbar = tqdm(range(args.training.warmup_epochs), desc="Warmup", leave=False, ncols=100)
            with open(f"{logging_dir}/loss_wsl.csv", "w") as f:
                f.write("epoch,loss,loss_2,val_acc,val_acc_2\n")
        for epoch in range(args.training.warmup_epochs):
            torch.cuda.empty_cache()
            train_loss_1 = baseline_train_iteration(
                model_1,
                optimizer_1,
                baseline_loss,
                baseline_penalty,
                labeled_dataloader,
                device=device,
            )
            train_loss_2 = baseline_train_iteration(
                model_2,
                optimizer_2,
                baseline_loss,
                baseline_penalty,
                labeled_dataloader_2,
                device=device,
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

        val_acc_best, val_acc_best_2 = 0., 0.
        for epoch in range(args.training.warmup_epochs, args.epochs):
            torch.cuda.empty_cache()
            prob_1, gmm_loss_1 = gmm_iteration(
                model_1,
                gmm_dataloader,
                dividemix_eval_loss,
                device=device,
            )
            prob_2, gmm_loss_2 = gmm_iteration(
                model_2,
                gmm_dataloader,
                dividemix_eval_loss,
                device=device,
            )
            prob_clean_1 = (prob_1 > args.training.p_threshold)
            prob_clean_2 = (prob_2 > args.training.p_threshold)

            # update labeled and unlabeled dataset (teaching each other)
            labeled_dataloader_2 = labeled_dataset.update(prob_1, prob_clean_1)
            unlabeled_dataloader_2 = unlabeled_dataset.update(prob_1, ~prob_clean_1)  # the remaining part
            labeled_dataloader = labeled_dataset_2.update(prob_2, prob_clean_2)
            unlabeled_dataloader = unlabeled_dataset_2.update(prob_2, ~prob_clean_2)

            train_loss_1 = train_iteration(
                model_1,
                model_2,
                optimizer_1,
                labeled_dataloader,
                unlabeled_dataloader,
                data_augment,
                augmented_samples=args.training.augmented_samples,
                augment_scale=(args.training.augment_mu, args.training.augment_std),
                sharpening_temp=args.training.sharpening_temp,
                alpha=args.training.alpha,
                num_labels=2,
                device=device,
            )
            train_loss_2 = train_iteration(
                model_2,
                model_1,
                optimizer_2,
                labeled_dataloader_2,
                unlabeled_dataloader_2,
                data_augment,
                augmented_samples=args.training.augmented_samples,
                augment_scale=(args.training.augment_mu, args.training.augment_std),
                sharpening_temp=args.training.sharpening_temp,
                alpha=args.training.alpha,
                num_labels=2,
                device=device,
            )
            val_acc_1 = val_iteration(
                model_1,
                val_dataloader,
                device=device,
            )
            val_acc_2 = val_iteration(
                model_2,
                val_dataloader,
                device=device,
            )
            if DIST_WRAPPER.rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f'{train_loss_1:.2f}', loss_2=f'{train_loss_2:.2f}',
                                 val_acc=f'{val_acc_1:.2f}', val_acc_2=f'{val_acc_2:.2f}')
                with open(f"{logging_dir}/loss_wsl.csv", "a") as f:
                    f.write(f"{epoch},{train_loss_1},{train_loss_2},{val_acc_1},{val_acc_2}\n")

                if epoch % args.training.save_interval == 0:
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_1_{epoch}.pt"))
                    torch.save(model_2.state_dict(), os.path.join(logging_dir, "checkpoints", f"model_2_{epoch}.pt"))

                    torch.save(gmm_loss_1, os.path.join(logging_dir, "checkpoints", f"gmm_loss_1_{epoch}.pt"))
                    torch.save(gmm_loss_2, os.path.join(logging_dir, "checkpoints", f"gmm_loss_2_{epoch}.pt"))

                if val_acc_1 > val_acc_best:
                    val_acc_best = val_acc_1
                    torch.save(model_1.state_dict(), os.path.join(logging_dir, "checkpoints", "best_1.pt"))
                if val_acc_2 > val_acc_best_2:
                    val_acc_best_2 = val_acc_2
                    torch.save(model_2.state_dict(), os.path.join(logging_dir, "checkpoints", "best_2.pt"))


if __name__ == "__main__":
    train()