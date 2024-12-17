''' thresh point functions '''
import numpy as np


def get_angle(point_data, is_relative=False):
    ''' Calculate absolute angle of the point data '''
    direction_vectors = point_data[:, 1, :] - point_data[:, 0, :]
    normalized_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
    angles = np.arctan2(normalized_vectors[:, 1], normalized_vectors[:, 0])
    res = np.unwrap(angles)
    return res - res[0] if is_relative else res


def derive_features(ts, args, add_ang=True, add_time=False):
    ''' generated features from original data '''
    new_ts = []
    for t in ts:
        if add_ang:  # add angle of current place
            ang = get_angle(t)
            ang_grad = np.gradient(ang)  # this could be a better option

            t = np.concatenate((t, ang, ang_grad), axis=2)
        if add_time:  # maybe find a better way to add time (currently repeat time for each tracked point)
            n = t.shape[0]
            time_gap = 1 / args.longest_ts  # give time a relative value
            time_array = np.linspace(0, time_gap * (n - 1), n).reshape(n, 1)  # shape (n, 1)
            time_array_expanded = np.expand_dims(time_array, axis=-1)  # shape (n, 1, 1)
            time_feature = np.tile(time_array_expanded, (1, t.shape[1], 1))  # shape (n, 3, 1)
            t = np.concatenate((t, time_feature), axis=2)
        new_ts.append(t)
    return new_ts
