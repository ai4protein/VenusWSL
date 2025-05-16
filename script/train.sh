#!/bin/bash

# train with Huggingface dataset
python src/train.py \
    data.dataset_type=hf \
    data.hf_dataset=AI4Protein/DeepLocBinary \
    data.sequence_column=aa_seq \
    data.id_column=name \
    data.label_column=label \
    training.task=binary \
    model.label_dim=2 \
    model.embedding_dim=64 \
    model.attn_dim=512 \
    training.baseline=true \
    training.warmup_epochs=10 \
    epochs=100 \
    batch_size=32 \
    optimizer.lr=0.001

# train with local dataset
python src/train.py \
    data.dataset_type=local \
    data.path_to_training_set=./data/train \
    data.path_to_validation_set=./data/val \
    data.path_to_teaching_set=./data/train \
    training.task=binary \
    model.label_dim=2 \
    model.embedding_dim=64 \
    model.attn_dim=512 \
    training.baseline=true \
    training.warmup_epochs=10 \
    epochs=100 \
    batch_size=32 \
    optimizer.lr=0.001

# train with WSL (non-baseline mode)
python src/train.py \
    data.dataset_type=hf \
    data.hf_dataset=AI4Protein/DeepLocBinary \
    data.sequence_column=aa_seq \
    data.id_column=name \
    data.label_column=label \
    training.task=binary \
    model.label_dim=2 \
    model.embedding_dim=64 \
    model.attn_dim=512 \
    training.baseline=false \
    training.warmup_epochs=10 \
    epochs=100 \
    batch_size=32 \
    optimizer.lr=0.001 