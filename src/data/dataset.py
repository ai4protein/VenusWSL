import copy
import pickle
from typing import Dict, Optional, List, Sequence, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DTYPE_MAPPING = {
    'label': torch.float32,
    'embedding': torch.float32,
}


class DataAugment:
    def __init__(self,
        mask: bool = False,
        noise: bool = False,
        interpolate: bool = False,
    ):
        self.mask = mask
        self.noise = noise
        self.interpolate = interpolate

    def __call__(
        self,
        embedding: torch.Tensor,
        **kwargs,
    ):
        if self.mask:
            return self.masked_embedding(embedding, **kwargs)
        elif self.noise:
            return self.noisy_embedding(embedding, **kwargs)
        elif self.interpolate:
            return self.interpolate_embedding(embedding, **kwargs)
        else:
            raise NotImplementedError

    @staticmethod
    def masked_embedding(
        embedding: torch.Tensor,
        threshold: float = 0.15,
    ):
        mask = torch.rand(embedding.shape[0]) > threshold
        masked_embedding = embedding * mask[:, None]
        return masked_embedding

    @staticmethod
    def noisy_embedding(
        embedding: torch.Tensor,
        mu: float = 0.0,
        std: float = 1.0,
    ):
        noise = torch.randn_like(embedding) * std + mu
        noisy_embedding = embedding + noise
        return noisy_embedding

    @staticmethod
    def interpolate_embedding(
        embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        alpha: float = 0.5,
    ):
        interpolated_embedding = alpha * embedding + (1 - alpha) * target_embedding
        return interpolated_embedding


class ProteinDataset(Dataset):
    def __init__(self,
        path_to_dataset: str,
        num_classes: int = 10,
        min_seq_len: int = 20,
        max_seq_len: int = 1024,
        batch_size: int = 32,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = False,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        assert path_to_dataset.endswith('.csv'), 'Dataset must be in CSV format'

        self.data = pd.read_csv(path_to_dataset)
        if max_seq_len > 0:
            self.data = self.data[self.data['aa_seq'].apply(lambda x: len(x) < max_seq_len)]
        if min_seq_len > 0:
            self.data = self.data[self.data['aa_seq'].apply(lambda x: len(x) < max_seq_len)]

        # sort by sequence length
        self.data = self.data.sort_values(by='aa_seq', key=lambda x: x.str.len(), ascending=False)
        self.data_to_iter = self.data.copy()
        self.pred = None

        # dataloader arguments
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def __len__(self):
        return len(self.data_to_iter)

    def __getitem__(self, idx):
        # sequence = self.data[idx]['aa_seq']
        label = self.data_to_iter['label'][idx]

        if len(label.split(',')) > 1:
            label_ = [0.] * self.num_classes
            for l in label.split(','):
                label_[int(l)] = 1.

        embedding_path = self.data_to_iter['embedding_path'][idx]
        if embedding_path.endswith('.pt'):
            embedding = torch.load(embedding_path, map_location='cpu')
        elif embedding_path.endswith('.pkl'):
            with open(embedding_path, 'rb') as f:
                embedding = pickle.load(f)

        data_object = {
            'label': label,
            'embedding': embedding,
            'mask': torch.ones(embedding.shape[0]),
        }
        if self.pred is not None:
            data_object['pred'] = self.pred[idx]

        data_object = self.map_to_tensors(data_object)
        return data_object

    def update(self, pred, w_clean):
        self.pred = pred[w_clean]
        self.data_to_iter = self.data[w_clean]  # w_clean should be a numpy array
        self.data_to_iter = self.data_to_iter.reset_index(drop=True)

        # reinitialize the dataset
        return self.get_dataloader()

    def get_dataloader(self):
        dataloader = DataLoader(
            self,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def map_to_tensors(chain_feats):
        chain_feats = {k: torch.as_tensor(v) for k, v in chain_feats.items()}
        # Alter dtype
        for k, dtype in DTYPE_MAPPING.items():
            if k in chain_feats:
                chain_feats[k] = chain_feats[k].type(dtype)
        return chain_feats


class BatchTensorConverter:
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, target_keys: Optional[List] = None):
        self.target_keys = target_keys

    def __call__(self, raw_batch: Sequence[Dict[str, object]]):
        B = len(raw_batch)
        # Only do for Tensor
        target_keys = self.target_keys \
            if self.target_keys is not None else [k for k, v in raw_batch[0].items() if torch.is_tensor(v)]
        # Non-array, for example string, int
        non_array_keys = [k for k in raw_batch[0] if k not in target_keys]
        collated_batch = dict()
        for k in target_keys:
            collated_batch[k] = self.collate_dense_tensors([d[k] for d in raw_batch], pad_v=0.0)
        for k in non_array_keys:  # return non-array keys as is
            collated_batch[k] = [d[k] for d in raw_batch]
        return collated_batch

    @staticmethod
    def collate_dense_tensors(samples: Sequence, pad_v: float = 0.0):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


