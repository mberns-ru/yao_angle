''' Training and testing loop '''
import torch
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix

from .models import random_dropping, CustomLRScheduler


def train_model(model, fold, train_loader, val_loader, device, main_args_ref):
    ''' Train the model and validate it on the validation set. '''
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    lr_hist = []
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=main_args_ref.wd)
    lr_scheduler = CustomLRScheduler(optimizer, main_args_ref)
    criterion = nn.CrossEntropyLoss()  # add weight if class imbalance
    for epoch in range(main_args_ref.epochs):
        model.train()  # train loop starts
        lr = lr_scheduler.step()
        lr_hist.append(lr)

        train_loss, total_train, correct_train = 0.0, 0, 0
        for batch_ts, batch_y, batch_len in train_loader:
            batch_ts = random_dropping(batch_ts, epoch, main_args_ref.drop_prob).to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            if main_args_ref.use_lstm:
                outputs = model(batch_ts, batch_len)
            else:
                outputs = model(batch_ts)
            loss = criterion(outputs, batch_y)

            # Calculate accuracy
            predicted_classes = torch.argmax(outputs.data, dim=1)
            true_classes = torch.argmax(batch_y, dim=1)
            correct_train += (predicted_classes == true_classes).sum().item()
            total_train += batch_y.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc = correct_train / total_train
        average_train_loss = train_loss / len(train_loader)
        model.eval()  # validation loop starts
        val_loss = 0
        total_val = 0
        correct_val = 0
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            for batch_ts, batch_y, batch_len in val_loader:
                batch_ts, batch_y = (
                    batch_ts.to(device),
                    batch_y.to(device),
                )
                if main_args_ref.use_lstm:
                    outputs = model(batch_ts, batch_len)
                else:
                    outputs = model(batch_ts)
                loss = criterion(outputs, batch_y)
                predicted_classes = torch.argmax(outputs.data, dim=1)
                true_classes = torch.argmax(batch_y, dim=1)
                # accumulate for f1 score
                val_labels.extend(true_classes.cpu().numpy())
                val_predictions.extend(predicted_classes.cpu().numpy())

                total_val += true_classes.size(0)

                correct_val += (predicted_classes == true_classes).sum().item()
                val_loss += loss.item()
        val_acc = correct_val / total_val
        average_val_loss = val_loss / len(val_loader)
        # keep track of losses and accuracies
        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_f1 = f1_score(val_labels, val_predictions, average='macro')
        if epoch % 10 == 0 or epoch == main_args_ref.epochs - 1:
            print(
                f'Fold {fold} -- Epoch {epoch}/{main_args_ref.epochs}  - Training Loss: {average_train_loss:.4f}, \
                Validation Loss: {average_val_loss:.4f}, Training Acc: {train_acc:.2f}, \
                Validation Acc: {val_acc:.2f}'
            )
    return (train_losses, val_losses, train_accuracies, val_accuracies), val_f1, lr_hist


def test_model(model, dataloader, class_num, main_args, device='cpu'):
    ''' Test the model on the test set. '''
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    total_test = 0
    correct_test = 0
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_ts, batch_y, batch_len in dataloader:
            batch_ts, batch_y = (
                batch_ts.to(device),
                batch_y.to(device),
            )
            if main_args.use_lstm:
                outputs = model(batch_ts, batch_len)
            else:
                outputs = model(batch_ts)  # Adjust based on your model
            loss = criterion(outputs, batch_y)
            predicted_classes = torch.argmax(outputs.data, dim=1)
            # If batch_y is already class indices
            true_classes = torch.argmax(batch_y, dim=1)

            # accuracy for each class
            correct_test += (predicted_classes == true_classes).sum().item()
            total_test += batch_y.size(0)
            test_loss += loss.item()
            test_predictions.extend(predicted_classes.cpu().numpy())
            test_targets.extend(true_classes.cpu().numpy())

    test_acc = correct_test / total_test
    average_test_loss = test_loss / len(dataloader)
    confusion_mtx = confusion_matrix(
        test_targets, test_predictions, labels=range(class_num))

    f1_score_list = f1_score(test_targets, test_predictions, average=None, labels=range(class_num))
    f1_score_dict = dict(enumerate(f1_score_list))
    print(f'Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_acc:.2f}')
    return average_test_loss, test_acc, confusion_mtx, f1_score_dict
