''' arg parser '''

import argparse


def parse_arguments():
    ''' argument parser '''
    parser = argparse.ArgumentParser(description='Train a model with cross-validation.')
    parser.add_argument('--data_path', type=str, default='../data/sound_control', help='Path to the data directory.')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4,  help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay for L2 regularization.')
    parser.add_argument('--k_folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--test_percent', type=float,  default=0.1, help='Percentage of data to use as the test set.',)
    parser.add_argument('--random_state', type=int, default=72, help='Random state for reproducibility.')
    parser.add_argument('--thresh_arg', type=int, default=-1, help='Frame Threshold for time series cutoff.')
    parser.add_argument('--training_type', type=str, default='TrialType', help='which column to use as label. `TrialType` or `Direction`')
    parser.add_argument('--use_lstm', type=bool, default=False, help='whether to use LSTM or Transformer')
    parser.add_argument('--test_only', type=bool, default=True, help='whether to use LSTM or Transformer')
    parser.add_argument('--hidden_size', type=int, default=32, help='hidden size for transformer')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers for transformer')
    parser.add_argument('--drop_prob', type=float, default=0.00, help='dropout probability for transformer')
    parser.add_argument('--template_anchor_csv_path', type=str, default='../data/anchors.csv', help='Path to the anchor csv file for camera distortion correction.')

    main_args = parser.parse_args()
    return main_args
