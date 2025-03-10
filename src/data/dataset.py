import pickle
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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
        max_seq_len: int = 1024,
    ):
        assert path_to_dataset.endswith('.csv'), 'Dataset must be in CSV format'

        self.data = pd.read_csv(path_to_dataset)
        if max_seq_len > 0:
            self.data = self.data[len(self.data['aa_seq']) <= max_seq_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]['aa_seq']
        label = self.data[idx]['label']

        data_object = {
            'sequence': sequence,
            'label': label,
        }

        if 'embedding_path' in self.data.columns:
            embedding_path = self.data[idx]['embedding_path']
            with open(embedding_path, 'rb') as f:
                embedding = pickle.load(f)

            data_object.update({
                'embedding': embedding,
            })

        return data_object



