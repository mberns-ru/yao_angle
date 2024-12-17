''' data prep from file to ml ready '''

from .data_cleaning import load_all_csv_and_h5_with_correction, data_filtering, ts_trim_at_thresh, get_y_tensor
from .feature_extraction import derive_features
# from .data_correction import extract_anchors, distortion_correction


def data_prep(raw_folder, freqs, types, args, thresh_arg, training_type, add_ang=True, add_time=True):
    '''
    This function processes data from file to ML ready
    '''
    df, ts = load_all_csv_and_h5_with_correction(raw_folder, args)  # load all data and merge to dataset
    df, ts = data_filtering(df, ts, only_correct=True, am_rates=freqs, trial_types=types)  # filter data by conditions
    args.longest_ts = max(x.shape[0] for x in ts)  # set longest ts as threshold
    ts = derive_features(ts, args, add_ang=add_ang, add_time=add_time)  # add angle and time features
    ts = ts_trim_at_thresh(ts, args, thresh_arg)  # cut off time series to a certain length
    ts_flat = [x.reshape(x.shape[0], -1) for x in ts]
    # --- to vector for training ---
    y = get_y_tensor(df, y_is=training_type)
    return y, ts_flat
