import datetime
import logging
import os
import warnings

import hydra
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
import rootutils

from src.data.dataset import DataAugment, ProteinDataset, BatchTensorConverter
from src.model.VenusWSL import PredictorPLM
from src.utils.ddp_utils import DIST_WRAPPER, seed_everything
from src.utils.training_utils import val_iteration

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base="1.3", config_path="../config", config_name="inference")
def inference(args: DictConfig):
    logging_dir = os.path.join(args.logging_dir, f"INF_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    if DIST_WRAPPER.rank == 0:
        # update logging directory with current time
        if not os.path.isdir(args.logging_dir):
            os.makedirs(args.logging_dir)
        os.makedirs(logging_dir)

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

    seed_everything(
        seed=args.seed,
        deterministic=args.deterministic,
    )

    test_dataset = ProteinDataset(
        path_to_dataset=args.data.path_to_dataset,
        max_seq_len=args.data.max_seq_len,
        batch_size=args.batch_size,
        collate_fn=BatchTensorConverter(),
        shuffle=False,
        num_workers=args.data.num_workers,
        pin_memory=args.data.pin_memory,
    )

    test_dataloader = test_dataset.get_dataloader()

    model = PredictorPLM(
        plm_embed_dim=args.model.embedding_dim,
        attn_dim=args.model.attn_dim,
        num_labels=args.model.label_dim,
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt_dir, map_location=device))

    acc = val_iteration(
        net=model,
        loader=test_dataloader,
        device=device,
    )

    logging.info(f"Accuracy: {acc}")


if __name__ == "__main__":
    inference()


