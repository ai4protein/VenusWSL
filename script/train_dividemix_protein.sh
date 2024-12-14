export HF_ENDPOINT=https://hf-mirror.com
warmup_epochs=10
lr=0.001
CUDA_VISIBLE_DEVICES=0 python train_dividemix_protein.py \
    --seed 42 \
    --batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_epochs 100 \
    --warmup_epochs ${warmup_epochs} \
    --lr ${lr} \
    --test_file test.csv \
    --output_dir ckpt/s42_we${warmup_epochs}_lr${lr}_adamw


export HF_ENDPOINT=https://hf-mirror.com
warmup_epochs=10
lr=0.001
CUDA_VISIBLE_DEVICES=0 python train_dividemix_protein.py \
    --seed 42 \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --num_epochs 100 \
    --warmup_epochs ${warmup_epochs} \
    --lr ${lr} \
    --test_file test.csv \
    --output_dir ckpt/esm2-650m_s42_we${warmup_epochs}_lr${lr}_adamw