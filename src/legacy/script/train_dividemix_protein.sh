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


warmup_epochs=10
lr=0.001
CUDA_VISIBLE_DEVICES=0 python train_dividemix_protein.py \
    --net_type plm_attn1d \
    --seed 42 \
    --batch_size 48 \
    --gradient_accumulation_steps 2 \
    --num_epochs 30 \
    --warmup_epochs ${warmup_epochs} \
    --lr ${lr} \
    --dataset_name tyang816/ProtSolM_ESMFold \
    --output_dir ckpt/test_plmattn1d_s42_we${warmup_epochs}_lr${lr}_adamw