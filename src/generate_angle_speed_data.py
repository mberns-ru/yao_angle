''' generate csv for angle data '''
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_prep.data_cleaning import h5_to_ts_list


def smooth_angles(angles, window_size):
    ''' Smooth the angle data using a moving average filter '''
    initial_angle = angles[0]
    adjusted_angles = angles - initial_angle
    pad_width = window_size - 1
    padded_angles = np.pad(adjusted_angles, (0, pad_width), mode='edge')
    smoothed_angles = np.convolve(padded_angles, np.ones(window_size) / window_size, mode='same')
    return smoothed_angles[:-pad_width] if pad_width > 0 else smoothed_angles


def find_sustained_below_threshold(data, window_size, threshold, min_duration=5):
    ''' Find the first index where the data stays below the threshold for at least min_duration '''
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    below_threshold = np.abs(moving_avg) < threshold
    sustained_indices = np.where(below_threshold)[0]
    if sustained_indices.size > 0:
        sustained_starts = np.where(np.convolve(below_threshold.astype(int), np.ones(min_duration, dtype=int), mode='valid') == min_duration)[0]
        if sustained_starts.size > 0:
            return sustained_starts[0] + window_size // 2 + min_duration // 2 - 1
    return -1  # Return -1 if no sustained region is found


def detect_plateau_edges(smoothed_angles, threshold=0.2, min_duration=5, window_size=20):
    ''' Detect edges of the plateau in smoothed angle data. '''
    gradients = np.gradient(smoothed_angles)
    left_index = find_sustained_below_threshold(gradients, window_size, threshold, min_duration)
    right_index = find_sustained_below_threshold(gradients[::-1], window_size, threshold, min_duration)
    if right_index != -1:
        right_index = len(gradients) - right_index - 1

    return left_index, right_index


def get_angle(extracted_points, relative=False):
    ''' Get angle from gerbil head to tail vector '''
    direction_vectors = extracted_points[:, 1, :] - extracted_points[:, 0, :]
    normalized_vectors = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]
    angles = np.arctan2(normalized_vectors[:, 1], normalized_vectors[:, 0])
    res = np.unwrap(angles)
    return res - res[0] if relative else res


def calculate_time_in_seconds(start_idx, end_idx, fps=60):
    ''' Convert frame count to seconds '''
    frame_count = end_idx - start_idx
    duration_seconds = frame_count / fps
    return duration_seconds


def get_angle_data(latency_val, extracted_points):
    ''' Get angle data for a single sample '''
    relative_angles = get_angle(extracted_points, True)
    absolute_angles = get_angle(extracted_points, False)
    smoothed_angles = smooth_angles(absolute_angles, window_size=10)
    _left_edge, right_edge = detect_plateau_edges(smoothed_angles)
    if pd.isna(latency_val):
        latency_val = _left_edge
    latency = int(latency_val)
    latency_frame = int((latency * 60) / 1000)
    tracked_time = calculate_time_in_seconds(latency_frame, right_edge)
    #print('time,start,end', tracked_time, latency_frame, right_edge)
    return relative_angles, tracked_time, latency_frame, right_edge


def get_angle_data_all(ts_list, df):
    ''' Get angle data for all the samples in the DataFrame '''
    corrected_angles_list = []
    decision_time_list = []
    start_frame = []
    end_frame = []
    for i, ts_sample in enumerate(ts_list):
        latency_val = df.iloc[i]['Latency']
        extracted_points = ts_sample[:, [0, 4], :]
        relative_angles, tracked_time, latency_frame, right_edge = get_angle_data(latency_val, extracted_points)
        corrected_angles_list.append(relative_angles)
        decision_time_list.append(tracked_time)
        start_frame.append(latency_frame)
        end_frame.append(right_edge)

    df['Corrected_Angles(Radians)'] = corrected_angles_list
    df['Decision Time (secs)'] = decision_time_list
    df['Decision start(frame number)'] = start_frame
    df['Decision end(frame number)'] = end_frame
    return df


def plot_angles(corrected_angles, latency_frame, csv_name):
    ''' Plot the corrected angles with a vertical line at the latency frame '''
    frame_numbers = range(1, len(corrected_angles) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, corrected_angles, marker='o', linestyle='-', color='blue')
    plt.axvline(x=latency_frame, color='red', linestyle='--', label='Special Frame')
    plt.text(latency_frame, max(corrected_angles) * 0.95, f'x = {latency_frame}', color='red', horizontalalignment='right')
    # Adding titles and labels
    plt.title('Angle Change wrt first frame vs. Frame Number')
    plt.xlabel('Frame Number')
    plt.ylabel('Angle Change (degrees)')
    plt.grid(True)
    plt.savefig(f'plots/{csv_name}')
    plt.close()


def plot_detected_plateau(angles, smoothed_angles, left_index, right_index, csv_name):
    ''' Plot the original, smoothed angles, and detected plateau edges '''
    plt.figure(figsize=(12, 6))
    plt.plot(angles, label='Original Angles', alpha=0.9)
    plt.plot(smoothed_angles, label='Smoothed Angles', linestyle='--', alpha=0.9)
    plt.plot(np.gradient(smoothed_angles), label='Smoothed Angle Gradients', linestyle='-', alpha=0.9)
    plt.axvline(x=left_index, color='r', linestyle='--', linewidth=1, label='Detected Left Edge')
    plt.axvline(x=right_index, color='g', linestyle='--', linewidth=1, label='Detected Right Edge')
    plt.legend()
    plt.title('Detected Plateau Edges with Sustained Edge Detection Logic')
    plt.xlabel('Index')
    plt.ylabel('Angle')
    plt.savefig(f'plots/{csv_name}')
    plt.close()

def parse_args():
    ''' parsing commandline arguments '''
    parser = argparse.ArgumentParser(description='Generate angle data for gerbil videos')
    parser.add_argument('--data_path', type=str, default='../../data', help='Path to the data directory')
    args = parser.parse_args()
    return args


def main():
    ''' Main function to generate angle data '''
    args = parse_args()
    data_path = Path(args.data_path)
    video_folders = [folder for folder in data_path.iterdir() if folder.is_dir()]
    for video_folder in video_folders:
        print(video_folder)
        h5_path = next(video_folder.glob('*.h5'), None)
        
        if h5_path is None or not h5_path.is_file():
            print(f"No valid .h5 file found in {video_folder}")
            continue  # Skip this folder if no valid h5 file
        
        csv_path = next(video_folder.glob('*.csv'), None)
        if csv_path is None or not csv_path.is_file():
            print(f"No valid .csv file found in {video_folder}")
            continue  # Skip this folder if no valid CSV file
        
        # Process the data
        df = pd.read_csv(csv_path)
        ts_list = h5_to_ts_list(Path(h5_path), df)
        new_df = get_angle_data_all(ts_list, df)
        
        # Save the modified DataFrame to a new CSV file
        new_filename = csv_path.stem + '_anglespeed' + csv_path.suffix
        new_csv_path = csv_path.parent / new_filename
        new_df.to_csv(new_csv_path, index=False)
        print(f"Saved generated data to {new_csv_path}")



if __name__ == '__main__':
    main()
