#!/bin/bash

data_base_dir=data/AI4Protein/DeepLocBinary
python src/train.py \
    epochs=100 \
    logging_dir='output/logs/DeepLocBinary' \
    training.baseline=True \
    training.task="binary" \
    data.path_to_teaching_set=${data_base_dir}/teaching_metadata.csv \
    data.path_to_training_set=${data_base_dir}/training_metadata_c.csv \
    data.path_to_validation_set=${data_base_dir}/validation_metadata.csv \
    data.embed_root=results/DeepLocBinary/esm2_t30_150M_UR50D \
    model.label_dim=2 \
    training.plm=esm2_t30_150M_UR50D \
    model.embedding_dim=640
    


python src/train.py \
    epochs=100 \
    logging_dir='output/logs/DeepLocBinary' \
    training.task="binary" \
    data.path_to_teaching_set=${data_base_dir}/teaching_metadata.csv \
    data.path_to_training_set=${data_base_dir}/training_metadata_c.csv \
    data.path_to_validation_set=${data_base_dir}/validation_metadata.csv \
    model.label_dim=2 \
    training.warmup_epochs=30 \
    training.plm=protbert \
    model.embedding_dim=1024

python src/train.py \
    epochs=100 \
    logging_dir='output/logs/DeepLocBinary' \
    training.baseline=True \
    training.task="binary" \
    data.path_to_teaching_set=${data_base_dir}/teaching_metadata.csv \
    data.path_to_training_set=${data_base_dir}/training_metadata_c.csv \
    data.path_to_validation_set=${data_base_dir}/validation_metadata.csv \
    model.label_dim=2 \
    training.plm=ankh \
    model.embedding_dim=1536


python src/train.py \
    epochs=100 \
    logging_dir='output/logs/DeepLocBinary' \
    training.task="binary" \
    data.path_to_teaching_set=${data_base_dir}/teaching_metadata.csv \
    data.path_to_training_set=${data_base_dir}/training_metadata_c.csv \
    data.path_to_validation_set=${data_base_dir}/validation_metadata.csv \
    model.label_dim=2 \
    training.warmup_epochs=30 \
    training.plm=ankh \
    model.embedding_dim=1536