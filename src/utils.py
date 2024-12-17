''' some utility functions. '''
import random

import torch
import numpy as np
import pandas as pd


def reset_random_seeds(seed_value=42):
    ''' reset all random seeds for reproducibility. '''
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def merge_record(csv_fp, rec_dict):
    ''' Merge the record to the csv file. '''
    if not csv_fp.exists():
        new_df = pd.DataFrame.from_dict(rec_dict)
        new_df.to_csv(csv_fp)
        return
    df = pd.read_csv(csv_fp, index_col=0)
    df.columns = df.columns.astype(int)
    new_df = pd.DataFrame.from_dict(rec_dict)
    merged_df = df.merge(new_df, left_index=True, right_index=True, how='left')
    merged_df.to_csv(csv_fp)
