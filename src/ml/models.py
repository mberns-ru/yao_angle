''' ML models for sequence classification '''
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    ''' LSTM model '''

    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense_1 = nn.Linear(hidden_size, 32)
        self.output_layer = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, time_series, lengths):
        ''' Forward pass '''
        # LSTM part of the network
        packed_time_series = pack_padded_sequence(time_series, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, (_, _) = self.lstm(packed_time_series)  # get LSTM output
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)  # Unpack packed
        lstm_out = lstm_out[range(len(lstm_out)), lengths - 1, :]  # Get last output

        # Feed forward part of the network
        dense_out = torch.relu(self.dense_1(lstm_out))

        # Output layer
        output = self.output_layer(dense_out)
        return output


class TransformerModel(nn.Module):
    ''' transformer model'''

    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=3, max_seq_length=100):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encodings = nn.Parameter(torch.zeros(max_seq_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            batch_first=True  # Set batch_first to True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        ''' Forward pass '''
        # Adding positional encodings to embeddings
        x = self.embedding(x) + self.positional_encodings[:x.size(1), :]
        x = self.transformer_encoder(x)
        # Considering only the output of the last time step for classification
        x = x[:, -1, :]
        x = self.output_layer(x)
        return x


class CustomLRScheduler:
    ''' Custom learning rate scheduler. '''

    def __init__(self, optimizer, main_args):
        self.optimizer = optimizer
        self.maintain_epoch = 160
        self.end_lr = 1e-5
        self.peak_lr = main_args.lr
        self.total_epochs = main_args.epochs
        self.current_epoch = 0

        self.decay_factor = lambda curr_epoch: (self.end_lr / self.peak_lr) ** (
            (curr_epoch - self.maintain_epoch) /
            (self.total_epochs - self.maintain_epoch)
        )  # custom decay steps

    def step(self):
        ''' Step the learning rate. '''
        if self.current_epoch < 80:  # fast warmup
            lr = 1e-3
        elif self.current_epoch < 120:
            lr = 5e-4
        elif self.current_epoch < self.maintain_epoch:
            lr = 1e-4
        else:
            lr = self.peak_lr * self.decay_factor(self.current_epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1
        return lr


def random_dropping(time_series, epoch, drop_prob=0.2):
    ''' Randomly drop some of the time steps. prevent overfitting.'''
    # Randomly drop some of the time steps
    if epoch < 0:  # experiment no dropping for first 150 epochs
        return time_series
    mask = torch.rand_like(time_series) > drop_prob
    return time_series * mask.float()
