''' Cross-validation for model evaluation. '''
import copy

import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from .models import LSTMModel, TransformerModel
from .train_test_loops import train_model


def cross_validate(dataset, device, main_args, collate_fn):
    ''' setup cross validation for better model evaluation. '''
    res = []
    best_model_ref = None
    best_val_f1 = 0
    f1_score_rec = []
    skf = StratifiedKFold(
        n_splits=main_args.k_folds, shuffle=True, random_state=main_args.random_state
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X=range(len(dataset)), y=dataset.target_array)
    ):

        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(
            dataset,
            batch_size=main_args.batch_size,
            sampler=train_subsampler,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=main_args.batch_size,
            sampler=val_subsampler,
            collate_fn=collate_fn,
        )

        if main_args.use_lstm:
            model = LSTMModel(
                input_size=main_args.input_size, hidden_size=128, num_layers=2,
                output_size=dataset.num_classes
            )
        else:
            model = TransformerModel(
                input_size=main_args.input_size,
                hidden_size=main_args.hidden_size,
                num_layers=main_args.num_layers,
                output_size=dataset.num_classes,
                max_seq_length=main_args.thresh_arg
            )
        model.to(device)
        if main_args.test_only:
            return None, model, None
        result, val_f1, _lr_hist = train_model(
            model,
            fold,
            train_loader,
            val_loader,
            device=device,
            main_args_ref=main_args
        )
        res.append(result)
        f1_score_rec.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_ref = copy.deepcopy(model)

    return get_avg_result(res), best_model_ref, f1_score_rec


def get_avg_result(train_res):
    ''' Process the results from cross-validation for plotting '''
    # results: [[train_losses], [val_losses], [train_accs], [val_accs]]
    num_epochs = len(train_res[0][0])
    # Initialize accumulators
    avg_train_losses, avg_train_accuracies = np.zeros(
        num_epochs), np.zeros(num_epochs)
    avg_val_losses, avg_val_accuracies = np.zeros(
        num_epochs), np.zeros(num_epochs)
    # Accumulate results from each fold
    for train_losses, val_losses, train_accuracies, val_accuracies in train_res:
        avg_train_losses += np.array(train_losses)
        avg_val_losses += np.array(val_losses)
        avg_train_accuracies += np.array(train_accuracies)
        avg_val_accuracies += np.array(val_accuracies)
    # Average the results across all folds
    avg_train_losses /= len(train_res)
    avg_val_losses /= len(train_res)
    avg_train_accuracies /= len(train_res)
    avg_val_accuracies /= len(train_res)
    return avg_train_losses, avg_val_losses, avg_train_accuracies, avg_val_accuracies
