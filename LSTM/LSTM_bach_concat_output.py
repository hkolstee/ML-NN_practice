import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

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
        #   ofc for now also train info leak into test by scaling here
        scaler = StandardScaler()
        scaler.fit(voices)
        scaled_voices = scaler.transform(voices)

        # initialize x data -> window_size amount of notes of 4 voices each per prediction
        self.x = np.zeros((self.nr_samples, window_size, self.nr_voices), dtype=np.float32)
        for i in range(self.x.shape[0]):
            self.x[i] = voices[i : i + window_size]
            # self.x[i] = scaled_voices[i : i + window_size]

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

# LSTM model with two conv layers
# the model is stateful, meaning the internal hidden state and cell state is passed
# into the model each batch and reset once per epoch
class LSTM_model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, batch_size):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        kernel_conv2 = 2
        c_out = 32
        lstm_input_size = (input_size - (kernel_conv2 - 1))

        # first conv layer
        self.conv2d_1 = nn.Conv2d(batch_size, c_out, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()

        # second conv layer
        c_out2 = c_out * 2
        self.conv2d_2 = nn.Conv2d(c_out, c_out2, kernel_size = 2)
        self.relu2 = nn.ReLU()

        self.lstm = nn.LSTM(c_out2 * lstm_input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        # self.softmax = nn.Softmax(1)

        # print("LSTM initialized with {} input size, {} hidden layer size, {} number of LSTM layers, and an output size of {}".format(input_size, hidden_size, num_layers, output_size))

    # reset hidden state and cell state, should be before each new sequence
    #   In our problem: every epoch, as it is one long sequence
    def reset_states(self):
        # hidden state and cell state for LSTM 
        self.hn = torch.zeros(self.num_layers,  1, self.hidden_size)
        self.cn = torch.zeros(self.num_layers, 1, self.hidden_size)

    def forward(self, input, stateful):
        # pass through first conv layer
        out = self.conv2d_1(input)
        out = self.relu1(out)

        # pass through second conv layer
        out = self.conv2d_2(out)
        out = self.relu2(out)

        # reshape for the lstm
        out = out.view(1, out.size(1), -1)

        # simple forward function
        # stateful = keep hidden states entire sequence length
        if stateful:
            out, (self.hn, self.cn) = self.lstm(out, (self.hn.detach(), self.cn.detach())) 
            out = self.linear(out[:,-1,:])
        else:
            hn = torch.zeros(self.num_layers,  1, self.hidden_size)
            cn = torch.zeros(self.num_layers, 1, self.hidden_size)
            out, (hn, cn) = lstm(out, (hn, cn)) 
            out = self.linear(out[:,-1,:])

        return out

def training(model, train_loader:DataLoader, test_loader:DataLoader, nr_epochs, optimizer, loss_func, stateful):
    # running loss per epoch (avg)
    running_loss_train = 0.
    running_loss_test = 0.

    for epoch in range(nr_epochs):
        # reset lstm hidden and cell state (stateful lstm = reset states once per sequence)
        # if not, reset automatically each forward call
        if stateful:
            model.reset_states()
        
        # train loop
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            # reset gradient function of weights
            optimizer.zero_grad()
            # forward
            prediction = model(inputs)
            # calculate loss
            loss = loss_func(prediction, labels)
            # backward, retain_graph = True needed for hidden lstm states
            loss.backward(retain_graph=True)
            # step
            optimizer.step()

            # tensorboard for visualization
            # writer.add_scalar("Loss/train", loss.item(), i + (epoch * len(train_loader)))
            running_loss_train += loss.item()
    
        # Test evaluation
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(test_loader):
                # forward pass
                prediction = model(inputs)
                # calculate loss
                test_loss = loss_func(prediction, labels)

                # tensorboard for visualization
                # writer.add_scalar("Loss/test", test_loss.item(), j + (epoch * len(test_loader)))
                running_loss_test += test_loss.item()

        # print training running loss and add to tensorboard
        print("Epoch:", epoch, "  Train loss:", running_loss_train/len(train_loader),
                                ", Test loss:", running_loss_test/len(test_loader))
        writer.add_scalar("Running train loss", running_loss_train/len(train_loader), epoch)
        writer.add_scalar("Running test loss", running_loss_test/len(test_loader), epoch)
        running_loss_train = 0
        running_loss_test = 0


    # tb writer flush
    writer.flush()

def createTrainTestDataloaders(dataset:NotesDataset, split_size, batch_size, shuffle, scale):
    # Train/test split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor((1 - split_size) * dataset_size))
    
    if shuffle:
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # create dataloaders
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size, num_workers=2)

    # if scale:
    #     scaler = StandardScaler()
    #     scaler.fit(train_loader.dataset.x)
    #     train_loader.dataset.x = scaler.transform(train_loader.dataset.x)
    
    return train_loader, test_loader

def main():
    # load data, 4 voices of instruments
    voices = np.loadtxt("F.txt")

    # remove starting silence, does not promote learning
    # data shape is (3816, 4) after
    voices = np.delete(voices, slice(8), axis=0)
    print(voices.shape)

    # Sliding window size used as input in model
    window_size = 48

    # create dataset based on window size where one window of timesteps
    #   will predict the subsequential single timestep
    dataset = NotesDataset(window_size, voices)

    # create data loaders for test and train
    split_size = 0.2
    batch_size = 1
    train_loader, test_loader = createTrainTestDataloaders(dataset, split_size, batch_size,shuffle=False, scale=True)
    
    features, labels = next(iter(train_loader))
    print("TRAIN: input size:", features.size(), 
          "- Output size:", labels.size(), 
          "- Samples:", len(train_loader), 
          "- TEST samples:", len(test_loader))

    # create model, nr_layers = number of sequential LSTM layers
    # input size = number of expected features
    # hidden size = number of features in hidden state 
    hidden_size = 46
    nr_layers = 2
    input_size = voices.shape[1]
    output_size = labels.size(1)
    lstm_model = LSTM_model(input_size, output_size, hidden_size, nr_layers, batch_size)

    # loss function and optimizer
    #   multi lable one hot encoded prediction only works with BCEwithlogitloss
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # training loop
    epochs = 150
    stateful = False
    training(lstm_model, train_loader, test_loader, epochs, optimizer, loss_func, stateful)

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True,linewidth=np.nan)

    main()

