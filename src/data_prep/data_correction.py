''' Helper functions for the affine transformation '''
import numpy as np


def get_affine_matrix(source, target):
    '''
    Computes the affine transformation matrix from source points to target points.
    Parameters:
    - source, target: numpy array of shape (n, 2) 
    Returns:
    - A 3x3 numpy array representing the affine transformation matrix. apply it on source you get target
    '''
    # Add ones to the last column of original and transformed points for affine transformation
    source_homogeneous = np.hstack((source, np.ones((source.shape[0], 1))))
    target_homogeneous = np.hstack((target, np.ones((target.shape[0], 1))))
    # Compute the affine matrix using linear algebra (A = T * inv(X) where T is the target and X is the source)
    affine_matrix, _, _, _ = np.linalg.lstsq(source_homogeneous, target_homogeneous, rcond=None)
    return affine_matrix


def apply_affine_matrix(time_series, affine_matrix):
    '''
    Applies the affine transformation matrix to a time series.
    Returns:
    - A numpy array of shape (n, k, 2) representing the transformed time series.
    '''
    n, k, _ = time_series.shape
    # reshape ts for transformation
    reshaped_time_series = time_series.reshape(n * k, 2)  # reshape the time series to a 2D array
    homogeneous_time_series = np.hstack((reshaped_time_series, np.ones((n * k, 1))))  # add third column for afine transformation
    # apply affine matrix
    transformed_homogeneous = np.dot(homogeneous_time_series, affine_matrix)  # Apply the transformation
    # shape ts back
    transformed_time_series = transformed_homogeneous[:, :2].reshape(n, k, 2)
    return transformed_time_series


def get_raw_xy(data, point):
    ''' fecth the x and y coordinates of a point from the data where both are not empty '''
    x_col = f'{point}.x'
    y_col = f'{point}.y'
    # Drop rows where either 'x' or 'y' is NaN
    clean_data = data[[x_col, y_col]].dropna()
    # Return as a numpy array
    return clean_data.to_numpy()


def bin_and_average(xy_data, bin_size=1.0):
    ''' bin the data and average the points in the most common bin (voting for truth) '''
    # Bin data
    x_bins = np.floor(xy_data[:, 0] / bin_size)
    y_bins = np.floor(xy_data[:, 1] / bin_size)
    # Find the bin with the most points
    bins, counts = np.unique(np.c_[x_bins, y_bins], axis=0, return_counts=True)
    max_bin_index = np.argmax(counts)
    most_common_bin = bins[max_bin_index]
    # Filter points that fall into the most common bin
    in_bin = (x_bins == most_common_bin[0]) & (y_bins == most_common_bin[1])
    filtered_points = xy_data[in_bin]
    # Calculate the average of the filtered points
    average_point = np.mean(filtered_points, axis=0)
    return average_point


def extract_anchors(df):
    ''' Extract the anchor points from the data and average them by binning '''
    anchors_list = []
    anchor_names = ['Nosepoke', 'Left_Food', 'Right_Food']
    for an in anchor_names:
        xy_raw = get_raw_xy(df, an)
        xy_avg = bin_and_average(xy_raw)
        anchors_list.append(xy_avg)
    anchor_np = np.array(anchors_list)
    return anchor_np


def distortion_correction(source_anchors, template_anchors, ts_source: np.ndarray) -> np.ndarray:
    ''' Find the affine matrix to transform source into shape of template, then apply to ts '''
    affine_matrix = get_affine_matrix(source_anchors, template_anchors)
    if ts_source.shape[2] != 2:  # make sure ts shape is correct
        raise ValueError('The time series should have shape (n, k, 2)')
    transformed_ts = apply_affine_matrix(ts_source, affine_matrix)
    return transformed_ts
