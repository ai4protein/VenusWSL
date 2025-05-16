#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path
import pickle
import gzip
import re

import torch
from transformers import (
    BertModel,
    BertTokenizer,
    EsmModel,
    AutoTokenizer,
    T5Tokenizer,
    T5EncoderModel
)
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

def get_model_and_tokenizer(model_name_or_path, model_dir=None):
    """Load model and tokenizer based on model type"""
    print("Loading: {}".format(model_name_or_path))
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
    
    if 'bert' in model_name_or_path:
        if model_dir is not None:
            tokenizer = BertTokenizer.from_pretrained(model_name_or_path, cache_dir=model_dir)
            model = BertModel.from_pretrained(model_name_or_path, cache_dir=model_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
            model = BertModel.from_pretrained(model_name_or_path)
    elif 'esm' in model_name_or_path:
        if model_dir is not None:
            model = EsmModel.from_pretrained(model_name_or_path, cache_dir=model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_dir)
        else:
            model = EsmModel.from_pretrained(model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    elif 't5' in model_name_or_path:
        if model_dir is not None:
            tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False, cache_dir=model_dir)
            model = T5EncoderModel.from_pretrained(model_name_or_path, cache_dir=model_dir)
        else:
            tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(model_name_or_path)
    elif 'ankh' in model_name_or_path:
        if model_dir is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_dir)
            model = T5EncoderModel.from_pretrained(model_name_or_path, cache_dir=model_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = T5EncoderModel.from_pretrained(model_name_or_path)
    else:
        raise ValueError(f"Unsupported model type: {model_name_or_path}")

    # Cast to full-precision if running on CPU
    if device == torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    return model, tokenizer

def read_fasta(fasta_path):
    """Read sequences from FASTA file"""
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                sequences[uniprot_id] = ''
            else:
                sequences[uniprot_id] += ''.join(line.split()).upper().replace("-", "")
    return sequences

def get_embeddings(seq_path,
                  emb_path,
                  model_name_or_path,
                  model_dir=None,
                  pooling_type=None,  # None, 'mean', 'max'
                  max_residues=4000,
                  max_seq_len=1000,
                  max_batch=100,
                  chunk_id=None,
                  total_chunks=None):
    """Extract embeddings from protein sequences using specified PLM
    
    Args:
        chunk_id: 当前处理的块ID (从0开始)
        total_chunks: 总块数
        pooling_type: 
            - None: return per-residue embeddings (seq_len x dim)
            - 'mean': return mean-pooled embedding (dim)
            - 'max': return max-pooled embedding (dim)
    """
    
    # Read sequences
    seq_dict = read_fasta(seq_path)
    
    # process specific chunk
    if chunk_id is not None and total_chunks is not None:
        total_seqs = len(seq_dict)
        chunk_size = total_seqs // total_chunks
        start_idx = chunk_id * chunk_size
        end_idx = start_idx + chunk_size if chunk_id < total_chunks - 1 else total_seqs
        
        # 将字典转换为列表，进行切片，再转回字典
        seq_items = list(seq_dict.items())[start_idx:end_idx]
        seq_dict = dict(seq_items)
        print(f"Processing chunk {chunk_id + 1}/{total_chunks} (sequences {start_idx + 1}-{end_idx})")
    
    model, tokenizer = get_model_and_tokenizer(model_name_or_path, model_dir)

    print('########################################')
    print('Example sequence: {}\n{}'.format(next(iter(seq_dict.keys())), next(iter(seq_dict.values()))))
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    # Process sequences
    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]), reverse=True)

    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    processed_count = 0
    skipped_count = 0
    
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict, desc="Processing sequences"), 1):
        # 检查embedding是否已存在
        embedding_path = os.path.join(emb_path, f"{pdb_id}.pkl.gz")
        if os.path.exists(embedding_path):
            skipped_count += 1
            continue

        # Preprocess sequence based on model type
        if 't5' in model_name_or_path:
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq = ' '.join(list(seq))  # Convert sequence to space-separated string
        seq_len = len(seq.split())  # Get actual sequence length after tokenization
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # Tokenize sequences
            token_encoding = tokenizer(seqs, 
                                     add_special_tokens=True, 
                                     padding="longest",
                                     return_tensors="pt")
            input_ids = token_encoding['input_ids'].to(device)
            attention_mask = token_encoding['attention_mask'].to(device)

            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                    
                    # Save embeddings
                    for batch_idx, identifier in enumerate(pdb_ids):
                        seq_len = seq_lens[batch_idx]
                        # Get the sequence embeddings (excluding padding)
                        seq_emb = features[batch_idx, :seq_len]  # [seq_len, hidden_dim]
                        
                        if pooling_type == 'mean':
                            # Mean pooling
                            emb = seq_emb.mean(dim=0)  # [hidden_dim]
                        elif pooling_type == 'max':
                            # Max pooling
                            emb = seq_emb.max(dim=0)[0]  # [hidden_dim]
                        else:  # None
                            # Return per-residue embeddings
                            emb = seq_emb  # [seq_len, hidden_dim]

                        if batch_idx == 0:
                            print("Embedded protein {} with length {} to emb. of shape: {}".format(
                                identifier, seq_len, emb.shape))

                        # Save embedding
                        embedding_path = os.path.join(emb_path, f"{identifier}.pkl.gz")
                        with gzip.open(embedding_path, "wb") as f:
                            pickle.dump({
                                'representation': emb.detach().cpu().numpy().squeeze(),
                                'pooling_type': pooling_type,
                                'sequence_length': seq_len
                            }, f)
                        processed_count += 1

            except RuntimeError:
                print(f"RuntimeError during embedding for {pdb_id} (L={seq_len}). Try lowering batch size.")
                continue

    end = time.time()
    print(f'\n############# STATS #############')
    print(f'Total time: {end - start:.2f}[s]')
    print(f'Processed sequences: {processed_count}')
    print(f'Skipped sequences: {skipped_count}')
    print(f'Time per processed sequence: {(end - start) / processed_count:.4f}[s]')
    print(f'Average sequence length: {avg_length:.2f}')
    return True

def get_embeddings_from_hf(dataset_name,
                          split=None,  # 改为可选参数
                          emb_path=None,
                          model_name_or_path=None,
                          model_dir=None,
                          pooling_type=None,
                          max_residues=4000,
                          max_seq_len=1000,
                          max_batch=100,
                          chunk_id=None,
                          total_chunks=None,
                          sequence_column='sequence',
                          id_column='id'):
    """Extract embeddings from Huggingface dataset
    
    Args:
        dataset_name: Huggingface dataset name
        split: Dataset split to use. If None, process all splits (train, validation, test)
        sequence_column: Name of the column containing protein sequences
        id_column: Name of the column containing protein IDs
    """
    # Load dataset
    if split is None:
        # 处理所有split
        splits = ['train', 'validation', 'test']
        for current_split in splits:
            try:
                print(f"\nProcessing split: {current_split}")
                os.makedirs(emb_path, exist_ok=True)
                
                dataset = load_dataset(dataset_name, split=current_split)
                process_dataset_split(
                    dataset=dataset,
                    emb_path=emb_path,
                    model_name_or_path=model_name_or_path,
                    model_dir=model_dir,
                    pooling_type=pooling_type,
                    max_residues=max_residues,
                    max_seq_len=max_seq_len,
                    max_batch=max_batch,
                    chunk_id=chunk_id,
                    total_chunks=total_chunks,
                    sequence_column=sequence_column,
                    id_column=id_column,
                    split_name=current_split
                )
            except Exception as e:
                print(f"Error processing split {current_split}: {str(e)}")
                continue
    else:
        # 处理单个split
        print(f"Loading dataset {dataset_name} split {split}")
        dataset = load_dataset(dataset_name, split=split)
        process_dataset_split(
            dataset=dataset,
            emb_path=emb_path,
            model_name_or_path=model_name_or_path,
            model_dir=model_dir,
            pooling_type=pooling_type,
            max_residues=max_residues,
            max_seq_len=max_seq_len,
            max_batch=max_batch,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            sequence_column=sequence_column,
            id_column=id_column,
            split_name=split
        )

def process_dataset_split(dataset,
                         emb_path,
                         model_name_or_path,
                         model_dir=None,
                         pooling_type=None,
                         max_residues=4000,
                         max_seq_len=1000,
                         max_batch=100,
                         chunk_id=None,
                         total_chunks=None,
                         sequence_column='sequence',
                         id_column='id',
                         split_name=None):
    """Process a single dataset split"""
    
    # 如果指定了分块，则只处理对应的块
    if chunk_id is not None and total_chunks is not None:
        total_seqs = len(dataset)
        chunk_size = total_seqs // total_chunks
        start_idx = chunk_id * chunk_size
        end_idx = start_idx + chunk_size if chunk_id < total_chunks - 1 else total_seqs
        dataset = dataset.select(range(start_idx, end_idx))
        print(f"Processing chunk {chunk_id + 1}/{total_chunks} (sequences {start_idx + 1}-{end_idx})")
    
    model, tokenizer = get_model_and_tokenizer(model_name_or_path, model_dir)

    print('########################################')
    print('Example sequence: {}\n{}'.format(dataset[0][id_column], dataset[0][sequence_column]))
    print('########################################')
    print('Total number of sequences: {}'.format(len(dataset)))

    # Process sequences
    seq_lens = [len(seq) for seq in dataset[sequence_column]]
    avg_length = np.mean(seq_lens)
    n_long = sum(1 for l in seq_lens if l > max_seq_len)

    print("Average sequence length: {:.2f}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    processed_count = 0
    skipped_count = 0
    
    for idx, example in enumerate(tqdm(dataset, desc=f"Processing {split_name} sequences")):
        pdb_id = example[id_column]
        seq = example[sequence_column]
        
        # 检查embedding是否已存在
        embedding_path = os.path.join(emb_path, f"{pdb_id}.pkl.gz")
        if os.path.exists(embedding_path):
            skipped_count += 1
            continue

        # Preprocess sequence based on model type
        if 't5' in model_name_or_path:
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq = ' '.join(list(seq))  # Convert sequence to space-separated string
        seq_len = len(seq.split())  # Get actual sequence length after tokenization
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or idx == len(dataset) - 1 or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # Tokenize sequences
            token_encoding = tokenizer(seqs, 
                                     add_special_tokens=True, 
                                     padding="longest",
                                     return_tensors="pt")
            input_ids = token_encoding['input_ids'].to(device)
            attention_mask = token_encoding['attention_mask'].to(device)

            try:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                    
                    # Save embeddings
                    for batch_idx, identifier in enumerate(pdb_ids):
                        seq_len = seq_lens[batch_idx]
                        # Get the sequence embeddings (excluding padding)
                        seq_emb = features[batch_idx, :seq_len]  # [seq_len, hidden_dim]
                        
                        if pooling_type == 'mean':
                            # Mean pooling
                            emb = seq_emb.mean(dim=0)  # [hidden_dim]
                        elif pooling_type == 'max':
                            # Max pooling
                            emb = seq_emb.max(dim=0)[0]  # [hidden_dim]
                        else:  # None
                            # Return per-residue embeddings
                            emb = seq_emb  # [seq_len, hidden_dim]

                        if batch_idx == 0:
                            print("Embedded protein {} with length {} to emb. of shape: {}".format(
                                identifier, seq_len, emb.shape))

                        # Save embedding with split name prefix
                        embedding_path = os.path.join(emb_path, f"{identifier}.pkl.gz")
                        with gzip.open(embedding_path, "wb") as f:
                            pickle.dump({
                                'representation': emb.detach().cpu().numpy().squeeze(),
                                'pooling_type': pooling_type,
                                'sequence_length': seq_len,
                                'split': split_name
                            }, f)
                        processed_count += 1

            except RuntimeError:
                print(f"RuntimeError during embedding for {pdb_id} (L={seq_len}). Try lowering batch size.")
                continue

    end = time.time()
    print(f'\n############# STATS for {split_name} #############')
    print(f'Total time: {end - start:.2f}[s]')
    print(f'Processed sequences: {processed_count}')
    print(f'Skipped sequences: {skipped_count}')
    print(f'Time per processed sequence: {(end - start) / processed_count:.4f}[s]')
    print(f'Average sequence length: {avg_length:.2f}')
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from protein sequences using various PLMs')
    
    # 添加新的参数组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help='Path to FASTA file containing protein sequence(s)')
    input_group.add_argument('--hf_dataset', type=str, help='Huggingface dataset name')
    
    parser.add_argument('--hf_split', type=str, default=None, 
                       help='Dataset split to use. If not specified, process all splits (train, validation, test)')
    parser.add_argument('--sequence_column', type=str, default='sequence', 
                       help='Name of the column containing protein sequences')
    parser.add_argument('--id_column', type=str, default='id', 
                       help='Name of the column containing protein IDs')
    
    parser.add_argument('-o', '--output', required=True, type=str, help='Path for saving the embeddings')
    parser.add_argument('--model_name_or_path', required=True, type=str, 
                        choices=[
                            'Rostlab/prot_bert', 'Rostlab/prot_bert_bfd',
                            'facebook/esm2_t30_150M_UR50D', 'facebook/esm2_t33_650M_UR50D', 
                            'Rostlab/prot_t5_xl_uniref50', 'Rostlab/prot_t5_xl_bfd',
                            'ElnaggarLab/ankh-large', 'ElnaggarLab/ankh-base'
                        ], 
                        help='Name of the pre-trained model to use')
    parser.add_argument('--model_dir', type=str, default=None, help='Path to directory holding the checkpoint for a pre-trained model')
    parser.add_argument('--pooling_type', type=str, 
                        choices=['mean', 'max', 'none'],
                        default='none',
                        help='Pooling strategy for sequence embeddings: mean, max, or none (per-residue)')
    parser.add_argument('--chunk_id', type=int, help='Chunk ID to process (0-based)')
    parser.add_argument('--total_chunks', type=int, help='Total number of chunks')
    args = parser.parse_args()

    emb_path = Path(args.output)
    model_dir = Path(args.model_dir) if args.model_dir is not None else None
    pooling_type = None if args.pooling_type == 'none' else args.pooling_type

    os.makedirs(emb_path, exist_ok=True)
    
    if args.hf_dataset:
        get_embeddings_from_hf(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            emb_path=emb_path,
            model_name_or_path=args.model_name_or_path,
            model_dir=model_dir,
            pooling_type=pooling_type,
            chunk_id=args.chunk_id,
            total_chunks=args.total_chunks,
            sequence_column=args.sequence_column,
            id_column=args.id_column
        )
    else:
        get_embeddings(
            seq_path=Path(args.input),
            emb_path=emb_path,
            model_name_or_path=args.model_name_or_path,
            model_dir=model_dir,
            pooling_type=pooling_type,
            chunk_id=args.chunk_id,
            total_chunks=args.total_chunks
        )

if __name__ == '__main__':
    main() 