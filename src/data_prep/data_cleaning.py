''' data cleaning stuff '''
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py

from .data_correction import extract_anchors, distortion_correction


def load_h5_matrix(fp: Path):
    '''Load a matrix from an h5 file.'''
    with h5py.File(fp, 'r') as f:
        tracks_matrix = f['tracks'][0].T  # pylint: disable=no-member
    return tracks_matrix  # (frame#, node#, coord)


def inter_2d(data: np.ndarray) -> np.ndarray:
    '''Interpolate a 2D numpy array using linear interpolation.
    Args:
    - data (np.ndarray): A 2D numpy array with 2 columns (x and y).
    Returns:
    - np.ndarray: The interpolated 2D numpy array.
    '''
    data_inter = np.empty_like(data)
    non_nan_idx = np.where(~np.isnan(data[:, 0]))[0]
    non_nan_values = data[non_nan_idx, :]
    # Interpolate the values at all indices
    for dim in range(2):
        data_inter[:, dim] = np.interp(range(data.shape[0]), non_nan_idx, non_nan_values[:, dim])
        # TODO savgol_filter(x, window_length=5, polyorder=2, axis=0)
    return data_inter


def tracks_to_chunks(df: pd.DataFrame, data: np.ndarray):
    ''' cut entire time series into chunks by trial '''
    chunks = []
    for _, row in df.iterrows():
        start_frame, end_frame = row['FrameStart'], row['FrameStop']
        chunk = data[start_frame:end_frame+1]  # slice the data array using start and end frames
        chunks.append(chunk)
    return chunks


def h5_to_ts_list(h5_path: Path, df: pd.DataFrame):
    ''' decode SLEAP h5 file into a list of time series '''
    tracks_matrix = load_h5_matrix(h5_path)
    # Interpolate tracks_matrix on axis=1
    inter_tracks = np.empty_like(tracks_matrix)
    total_chunks = tracks_matrix.shape[1]
    for i in range(total_chunks):
        inter_tracks[:, i, :] = inter_2d(tracks_matrix[:, i, :])
    chunks = tracks_to_chunks(df, inter_tracks)  # cut time series by trial
    return chunks


def _find_file_by_keyword(target_directory: Path, keyword: str, suffix: str = None):
    ''' Find file by keyword and suffix in a directory '''
    keyword = keyword.lower()
    for file in target_directory.iterdir():
        keyword_match = keyword in file.stem.lower()
        suffix_match = (suffix is None or file.suffix.lower() == '.' + suffix.lower())
        if keyword_match and suffix_match:
            return file
    return None


def _get_file_paths(curr_dir: Path):
    ''' get files for each experiment '''
    correction_keyword = 'correction'  # change this to the keyword of the correction file
    h5_path = next(curr_dir.glob('*.h5'), None)
    all_csv_files = [file for file in curr_dir.glob('*.csv')]
    correction_path = _find_file_by_keyword(curr_dir, correction_keyword, 'csv')
    csv_no_correction = [f for f in all_csv_files if f != correction_path]
    csv_path = csv_no_correction[0] if csv_no_correction else None
    return h5_path, csv_path, correction_path


def load_all_csv_and_h5_with_correction(raw_folder: Path, args):
    ''' read all csv and h5 files with a one-to-one correspondence '''
    ts_list_collections = []
    df_collections = []
    folders = [x for x in raw_folder.iterdir() if x.is_dir()]
    for folder in folders:
        h5_path, csv_path, correction_path = _get_file_paths(folder)
        template_anchors = extract_anchors(pd.read_csv(args.template_anchor_csv_path))
        source_anchors = extract_anchors(pd.read_csv(correction_path))
        # read file
        if csv_path is None or h5_path is None:
            continue  # Skip if no CSV file
        df = pd.read_csv(csv_path)
        ts_list = h5_to_ts_list(h5_path, df) if h5_path is not None else []
        ts_list_corrected = [distortion_correction(source_anchors, template_anchors, ts) for ts in ts_list]
        if len(df) != len(ts_list):
            raise ValueError(f'csv and ts not same amount: {folder}')
        # Flip the y-axis of the time series (from time n to time 0 -> time 0 to time n)
        ts_list_flipped = [np.flip(x, axis=0) for x in ts_list_corrected]
        ts_list_collections.extend(ts_list_flipped)
        df_collections.append(df)
    # Concatenate all DataFrames ensuring a continuous index
    try:
        df_all = pd.concat(df_collections, ignore_index=True)
    except ValueError as e:
        print(f'ValueError: {e}')
    return df_all, ts_list_collections


def data_filtering(df, ts, only_correct=True, trial_types=None, am_rates=None):
    '''
    Filter csv and ts by conditions
    return filtered df and ts
    '''
    print(f'Init: df shape: {df.shape}, ts length: {len(ts)}')

    # Handle 'Score' column (filter only correct trials)
    if 'Score' in df.columns:
        if only_correct:
            df.reset_index(drop=True, inplace=True)
            filter_idx = df[df['Score'] == 'Correct'].index
            df = df.loc[filter_idx]
            ts = [ts[i] for i in filter_idx.tolist()]

    # Handle 'AMRate' column (filter by required AM rates)
    if 'AMRate' in df.columns:
        if am_rates is not None:
            df.reset_index(drop=True, inplace=True)

            def _float_isin(x):
                return any(abs(x - v) <= 0.01 for v in am_rates)  # float ==

            filter_idx = df[df['AMRate'].apply(_float_isin)].index
            df = df.loc[filter_idx]
            ts = [ts[i] for i in filter_idx.tolist()]

    # Handle 'TrialType' column # (filter by required trial types)
    if 'TrialType' in df.columns:
        if trial_types is not None:
            df.reset_index(drop=True, inplace=True)
            filter_idx = df[df['TrialType'].isin(trial_types)].index
            df = df.loc[filter_idx]
            ts = [ts[i] for i in filter_idx.tolist()]

    return df, ts


def get_y_tensor(df, y_is='TrialType'):
    ''' get y tensor (for training) from dataframe '''
    if y_is == 'TrialType':
        # Map each TrialType category to a unique integer
        trial_type_mapping = {'Aud': 0, 'Vis': 1, 'Av': 2}
        df['TrialType_Int'] = df['TrialType'].map(trial_type_mapping)
        y = torch.tensor(df['TrialType_Int'].values, dtype=torch.long)  # Use dtype=torch.long for integer labels
    elif y_is == 'Direction':
        # Mapping Direction to binary values and then to a tensor
        df['Direction'] = df['Direction'].map({'Left': 0, 'Right': 1})
        y = torch.tensor(df['Direction'].values, dtype=torch.long)  # Use dtype=torch.long for integer labels
    return y


def ts_trim_at_thresh(ts, args, thresh_arg):
    ''' cut off time series to a certain length '''
    if thresh_arg == -1:
        args.thresh_arg = args.longest_ts
        return ts
    return [x[:thresh_arg] for x in ts]
