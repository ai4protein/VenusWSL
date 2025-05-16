import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset


def inject_label_noise(df, noisy_ratio=.5, num_classes=2, multi_label=False):
    """
    Injects label noise into the DataFrame.
    :param df: DataFrame containing the data.
    :param noise_level: Fraction of labels to be changed.
    :return: DataFrame with injected label noise.
    """
    # Randomly select a fraction of the labels to be changed
    num_samples = len(df)
    num_noisy_labels = int(num_samples * noisy_ratio)
    noisy_indices = df.sample(num_noisy_labels, random_state=42).index

    # Change the labels of the selected samples
    if not multi_label:
        df.loc[noisy_indices, 'label'] = np.random.randint(0, num_classes, size=num_noisy_labels)
    else:
        mean_label_length = df['label'].str.split(',').apply(len).mean()
        sampled_label_length = np.random.poisson(mean_label_length, size=num_noisy_labels)
        sampled_label_length[sampled_label_length <= 2] = 2
        sampled_label = [np.random.randint(0, num_classes, size=crt_label_length) for crt_label_length in sampled_label_length]
        sampled_label_string = [','.join([str(i) for i in crt_label]) for crt_label in sampled_label]
        df.loc[noisy_indices, 'label'] = sampled_label_string
    return df


def inject_gaussian_noise(df, noisy_ratio=.5):
    """
    Injects Gaussian noise into the DataFrame.
    :param df: DataFrame containing the data.
    :param noisy_ratio: Fraction of labels to be changed.
    :return: DataFrame with injected label noise.
    """
    # Randomly select a fraction of the labels to be changed
    num_samples = len(df)
    num_noisy_labels = int(num_samples * noisy_ratio)
    noisy_indices = df.sample(num_noisy_labels, random_state=42).index

    # Change the labels of the selected samples
    std = df['label'].std()
    df.loc[noisy_indices, 'label'] += np.random.normal(0, std, size=num_noisy_labels)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_dataset', type=str, default='AI4Protein/DeepLocBinary')
    parser.add_argument('--noisy_ratio', type=float, default=0.5)
    parser.add_argument('--is_label_noise', action='store_true')
    parser.add_argument('--output_dir', type=str, default='data/AI4Protein')
    args = parser.parse_args()
    
    dataset_name = args.hf_dataset.split('/')[-1]
    print(f'>>> Processing {dataset_name}')
    
    os.makedirs(os.path.join(args.output_dir, dataset_name), exist_ok=True)
    # ['ec', 'go_mf', 'go_cc', 'go_bp', 'sol_binary', 'loc_binary', 'loc_multi_class', 'loc_multi_label', 'thermo', 'temp_opt']
    regression_tasks = ['Thermostability', 'DeepET_Topt']
    classification_tasks = ['EC', 'GO_MF', 'GO_CC', 'GO_BP', 'DeepSol', 'DeepLocBinary', 'DeepLocMulti', 'DeepLoc2Multi']
    num_classes = [585, 489, 320, 1943, 2, 2, 10, 10]
    multi_label = [True, True, True, True, False, False, False, True]

    if dataset_name in classification_tasks:
        dataset = load_dataset(args.hf_dataset)
        # convert to pandas dataframe
        train_set = dataset['train'].to_pandas()
        valid_set = dataset['validation'].to_pandas()
        test_set = dataset['test'].to_pandas()

        n_class = num_classes[classification_tasks.index(dataset_name)]
        ml = multi_label[classification_tasks.index(dataset_name)]

        # sample 50% from train set
        cropped_train_set = train_set.sample(frac=0.5, random_state=42)
        noisy_train_set = train_set[~train_set.index.isin(cropped_train_set.index)]

        if args.is_label_noise:
            # inject label noise
            noisy_train_set = inject_label_noise(noisy_train_set, noisy_ratio=0.5, num_classes=n_class, multi_label=ml)

        # save updated datasets
        cropped_train_set.to_csv(os.path.join(args.output_dir, dataset_name, 'teaching_metadata.csv'), index=False)
        noisy_train_set.to_csv(os.path.join(args.output_dir, dataset_name, 'training_metadata_c.csv'), index=False)
        valid_set.to_csv(os.path.join(args.output_dir, dataset_name, 'validation_metadata.csv'), index=False)
        test_set.to_csv(os.path.join(args.output_dir, dataset_name, 'testing_metadata.csv'), index=False)

    if dataset_name in regression_tasks:
        dataset = load_dataset(args.hf_dataset)
        train_set = dataset['train'].to_pandas()
        valid_set = dataset['validation'].to_pandas()
        test_set = dataset['test'].to_pandas()
        
        # sample 50% from train set
        cropped_train_set = train_set.sample(frac=0.5, random_state=42)
        noisy_train_set = train_set[~train_set.index.isin(cropped_train_set.index)]
            
        if args.is_label_noise:
            # inject label noise
            noisy_train_set = inject_gaussian_noise(noisy_train_set, noisy_ratio=0.5)

        # save updated datasets
        cropped_train_set.to_csv(os.path.join(args.output_dir, dataset_name, 'teaching_metadata.csv'), index=False)
        noisy_train_set.to_csv(os.path.join(args.output_dir, dataset_name, 'training_metadata_c.csv'), index=False)
        valid_set.to_csv(os.path.join(args.output_dir, dataset_name, 'validation_metadata.csv'), index=False)
        test_set.to_csv(os.path.join(args.output_dir, dataset_name, 'testing_metadata.csv'), index=False)
        
    # # process sol_binary
    # train_set = pd.read_csv(os.path.join('sol_binary', 'train.csv'))
    # valid_set = pd.read_csv(os.path.join('sol_binary', 'valid.csv'))
    # test_set = pd.read_csv(os.path.join('sol_binary', 'test.csv'))

    # cropped_test_set = test_set.sample(frac=0.5, random_state=42)
    # noisy_test_set = test_set[~test_set.index.isin(cropped_test_set.index)]

    # cropped_test_set.to_csv(os.path.join('sol_binary', 'teaching_metadata.csv'), index=False)
    # noisy_test_set.to_csv(os.path.join('sol_binary', 'testing_metadata.csv'), index=False)
    # train_set.to_csv(os.path.join('sol_binary', 'training_metadata.csv'), index=False)
    # valid_set.to_csv(os.path.join('sol_binary', 'validation_metadata.csv'), index=False)