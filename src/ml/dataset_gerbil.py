''' Gerbil decision making dataset for ML model '''
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class GerbilDataset(Dataset):
    """Xt, X and y dataset."""

    def __init__(self, time_series_chunks, target_array, num_classes):
        filtered_data = [
            (ts, target)
            for ts, target in zip(time_series_chunks, target_array)
            if len(ts) > 0
        ]
        self.time_series_chunks, self.target_array = zip(*filtered_data)
        self.target_array = np.array(self.target_array)
        self.num_classes = num_classes

    def __len__(self):
        return len(self.time_series_chunks)

    def __getitem__(self, idx):
        ts_chunk = torch.tensor(
            self.time_series_chunks[idx], dtype=torch.float32
        )
        target = torch.tensor(
            self.target_array[idx], dtype=torch.long
        )  # torch.tensor(, dtype=torch.float32)
        return ts_chunk, target


def create_datasets(time_series_chunks, target_array, main_args):
    '''Split data into train and test sets.'''
    features_ts_train, features_ts_test, labels_train, labels_test = train_test_split(
        time_series_chunks,
        target_array,
        test_size=main_args.test_percent,
        random_state=main_args.random_state,
    )
    # Assuming GerbilDataset is your custom dataset class
    num_classes = len(np.unique(target_array))
    train_ds = GerbilDataset(
        features_ts_train, labels_train, num_classes)
    test_ds = GerbilDataset(features_ts_test, labels_test, num_classes)

    return train_ds, test_ds


def custom_collate(batch, num_classes):
    ''' Custom collate function to prep input data for network ingestion. '''
    time_series, targets = zip(*batch)
    lengths = torch.tensor([len(ts) for ts in time_series])
    time_series_padded = pad_sequence(time_series, batch_first=True)

    # Ensure targets are in long format
    targets = torch.tensor(targets, dtype=torch.long)
    targets_one_hot = torch.eye(num_classes)[targets]
    return time_series_padded, targets_one_hot, lengths
