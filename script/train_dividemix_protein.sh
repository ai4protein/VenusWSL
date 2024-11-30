warmup_epochs=10
lr=0.002

python dividemix_protein.py \
    --seed 42 \
    --num_epochs 100 \
    --warmup_epochs ${warmup_epochs} \
    --lr ${lr} \
    --gpuid 3 \
    --test_file test.csv \
    --output_dir ckpt/s42_we${warmup_epochs}_lr${lr}_adamw
