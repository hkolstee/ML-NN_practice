import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

# returns concatenated onehot encoding for each note 
def one_hot_encode(y: np.ndarray, voices: np.ndarray) -> np.ndarray:
    # unique set of notes in the voice
    unique_notes = np.unique(voices)

    # initialize return array
    encoded = np.zeros((y.shape[0], y.shape[1] * len(unique_notes)), dtype=np.float32)
    
    # one hot encode each note
    for timestep, notes in enumerate(y):
        for voice, note in enumerate(notes):
            one_hot_location = np.nonzero(unique_notes == note)[0][0]
            encoded[timestep][one_hot_location + voice * len(unique_notes)] = 1 

    return encoded

class NotesDataset(Dataset):
    def __init__(self, window_size: int, voices: np.ndarray):
        # nr of samples, and nr of voices
        self.nr_samples = voices.shape[0] - window_size
        self.nr_voices = voices.shape[1]

        # scale x (-> time information leak here but for now whatever)
        scaler = StandardScaler()
        scaler.fit(voices)
        scaled_voices = scaler.transform(voices)

        # initialize x data -> window_size amount of notes of 4 voices each per prediction
        self.x = np.zeros((self.nr_samples, window_size, self.nr_voices), dtype=np.float32)
        for i in range(self.x.shape[0]):
            self.x[i] = scaled_voices[i : i + window_size]

        # initialize y data -> 4 following target notes per time window 
        self.y = np.zeros((self.nr_samples, self.nr_voices), dtype = np.float32)
        for j in range(self.y.shape[0]):
            self.y[j] = voices[j + window_size]

        # one hot encode target tensor
        self.y = one_hot_encode(self.y, voices)

        # create tensors
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nr_samples

class LSTM_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(1)

        print("LSTM initialized with {} input size, {} hidden layer size, {} number of LSTM layers, and an output size of {}".format(input_size, hidden_size, num_layers, output_size))

    def forward(self, input):
        # hidden state and cell state (only for new step, other hidden states are in LSTM itself)
        hn = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        cn = torch.zeros(self.num_layers, input.size(0), self.hidden_size)

        # simple forward function
        out, (hn, cn) = self.lstm(input, (hn, cn)) 
        out = self.linear(out[:,-1,:])

        return out

def training(model, dataloader:DataLoader, nr_epochs, optimizer, loss_func):
    # get total samples from dataset
    total_samples = len(dataloader.dataset)
    nr_batches = len(dataloader)

    for epoch in range(nr_epochs):
        for index, (inputs, labels) in enumerate(tqdm(dataloader)):
            # reset gradient function of weights
            optimizer.zero_grad()
            # forward
            prediction = model(inputs)
            # calculate loss
            loss = loss_func(prediction, labels)
            # backward
            loss.backward()
            # step
            optimizer.step()

            # add loss to tensorboard
            writer.add_scalar("Loss/train", loss, (index + epoch * len(dataloader.dataset)))

        print("Epoch:", epoch, "  Loss:", loss.item() )

    # tb writer flush
    writer.flush()

def main():
    # load data, 4 voices of instruments
    voices = np.loadtxt("F.txt")

    # remove starting silence, does not promote learning
    # data shape is (3816, 4) after
    voices = np.delete(voices, slice(8), axis=0)
    print(voices.shape)

    # Sliding window size used as input in model
    window_size = 32

    # create dataset based on window size where one window of timesteps
    #   will predict the subsequential single timestep
    dataset = NotesDataset(window_size, voices)

    # create dataloader
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=2)
    features, labels = next(iter(dataloader))
    print("input size:", features.size(), "- Output size:", labels.size(), "- Samples:", len(dataloader.dataset))
    # print(dataset.x)
    # print(dataset.y)

    # create model, nr_layers = number of sequential LSTM layers
    # input size = number of expected features
    # hidden size = number of features in hidden state 
    hidden_size = 32
    nr_layers = 1
    input_size = voices.shape[1]
    output_size = 184
    lstm_model = LSTM_model(input_size, output_size, hidden_size, nr_layers)

    # loss function and optimizer
    #   multi lable one hot encoded prediction only works with BCEwithlogitloss
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # training loop
    training(lstm_model, dataloader, 100, optimizer, loss_func)
    

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True,linewidth=np.nan)

    main()

