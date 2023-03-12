import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from LSTM_bach import LSTM_model, NotesDataset


def calculateNotes():
    pass

def predictNextNotes(input, steps, lstm_model):
    sigmoid = nn.Sigmoid()
    # for i in range(steps):
    print(input.shape)
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    output = lstm_model(input, stateful=False)
    output = sigmoid(output)
    print(output)
    

def main():
    # define parameters used here
        # sliding window size
    window_size = 16
    hidden_size = 16
    conv_channels = 16
    input_size = 4
    output_size = 98
    num_layers = 1
        # train/test split
    split_size = 0.1
    batch_size = 1

    # initialize model
    model = LSTM_model(input_size, output_size, hidden_size, num_layers, batch_size, conv_channels)
    model.load_state_dict(torch.load("models/model161616_useless.pth"))

    # load data, 4 voices of instruments
    voices = np.loadtxt("F.txt")

    # Train/test split (needed for correct scaling of new data)
    dataset_size = len(voices[:,])
    indices = list(range(dataset_size))
    split = int(np.floor((1 - split_size) * dataset_size))
    train_indices = indices[:split]
    # create split in data
    train_voices = voices[train_indices, :]

    # fit the scaler to the train data
    scaler = StandardScaler()
    scaler.fit(train_voices)
    # scale voices
    voices = scaler.transform(voices)
    train_voices = scaler.transform(train_voices)

    # take last sliding window in data and infer from there
    input = train_voices[-16:]
    steps = 1000
    predictNextNotes(input, steps, model)

if __name__ == '__main__':
    torch.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(precision=3)
    torch.set_printoptions(sci_mode=False)
    main()