''' Main script for training and testing models. '''
from pathlib import Path
from functools import partial
# first party modules
import pandas as pd
import torch
from torch.utils.data import DataLoader

# our modules
from .data_prep import data_prep
from .ml.dataset_gerbil import custom_collate, create_datasets
from .ml.cross_validation import cross_validate
from .ml.train_test_loops import test_model

from .plots import plot_cross_validation_results, plot_matrix
from .args import parse_arguments
from .utils import reset_random_seeds


if __name__ == '__main__':
    args = parse_arguments()
    types = ['Aud', 'Vis', 'Av']
    freqs = [4, 5.26, 6.92, 9.12, 12]
    raw_folder = Path(args.data_path)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    reset_random_seeds(args.random_state)
    f1_results = {}

    y, ts = data_prep(raw_folder, freqs, types, args, thresh_arg=args.thresh_arg,
                      training_type=args.training_type, add_ang=True, add_time=True)

    # ------For loop should start here------
    for thresh_arg in (-1, ):  # , 20, 30, 40, 50, 60, 70, 80, 90, 100, -1]:
        args.thresh_arg = thresh_arg
        args.input_size = ts[0].shape[-1]
        # instead of split into train val and test, now we delay the train val split to cross validation
        train_dataset, test_dataset = create_datasets(ts, y, args)
        collate_func = partial(custom_collate, num_classes=train_dataset.num_classes)

        results, best_model, f1_scores = cross_validate(train_dataset, device=DEVICE, main_args=args, collate_fn=collate_func)

        fn = f'./results/thresh_frame_{args.thresh_arg}'
        model_path = Path('./models') / f'thresh_frame_{args.thresh_arg}.pt'
        if args.test_only:
            best_model.load_state_dict(torch.load(model_path))
        else:
            torch.save(best_model.state_dict(), model_path)
            plot_cross_validation_results(*results, fn)

        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_func)
        _, _, confusion, f1_dict = test_model(best_model, test_loader, train_dataset.num_classes, main_args=args, device=DEVICE)
        f1_results[args.thresh_arg] = f1_dict

        precision = confusion / confusion.sum(axis=1)[:, None]
        plot_matrix(precision, types, fn)
    # save f1_results to csv
    csv_path = Path('./results/f1_results.csv')
    df_update = pd.DataFrame.from_dict(f1_results)
    # merge_record(csv_path, df_update)
    # save to csv, overwrite the old one
