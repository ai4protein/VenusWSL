# model_name_or_path:
#   Rostlab/prot_bert, Rostlab/prot_bert_bfd
#   facebook/esm2_t30_150M_UR50D, facebook/esm2_t33_650M_UR50D
#   Rostlab/prot_t5_xl_uniref50, Rostlab/prot_t5_xl_bfd
#   ElnaggarLab/ankh-large, ElnaggarLab/ankh-base

CUDA_VISIBLE_DEVICES=0 python get_plm_embed.py \
    --input dataset/all_seqs.fasta \
    --output results/all_seqs_ankh-large \
    --model_name_or_path ElnaggarLab/ankh-large \
    --chunk_id 0 \
    --total_chunks 1

CUDA_VISIBLE_DEVICES=0 python get_plm_embed.py \
    --input dataset/all_seqs.fasta \
    --output results/all_seqs_prot_bert \
    --model_name_or_path Rostlab/prot_bert \
    --chunk_id 0 \
    --total_chunks 1

# handle all splits
python get_plm_embed.py \
    --hf_dataset AI4Protein/DeepLocBinary \
    --output results/DeepLocBinary/esm2_t30_150M_UR50D \
    --model_name_or_path facebook/esm2_t30_150M_UR50D \
    --id_column name \
    --sequence_column aa_seq

# handle specific split
python get_plm_embed.py \
    --hf_dataset AI4Protein/DeepLocBinary \
    --hf_split train \
    --output results/DeepLocBinary/train_esm2_t30_150M_UR50D \
    --model_name_or_path facebook/esm2_t30_150M_UR50D \
    --pooling_type mean \
    --id_column name \
    --sequence_column aa_seq