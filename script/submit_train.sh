

lr=(0.001 0.0001)
warmup_epochs=(1 2 3 4 5 6 7 8 9 10)

for lr in ${lr[@]}; do
    for warmup_epochs in ${warmup_epochs[@]}; do
        qsub -v lr=${lr},warmup_epochs=${warmup_epochs} -N wsl_we${warmup_epochs}_lr${lr} train_dividemix_protein.pbs
    done
done


for i in {114965..115000}
do
    qdel ${i}.fs
done