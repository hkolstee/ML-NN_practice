import numpy as np
import math
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

from LSTM_bach import LSTM_model, NotesDataset

# to find float index in unique float list of standar scaled array
# works also for ints when not scaled
def uniqueLocation(uniques, note):
    for index, unique in enumerate(uniques):
        if (math.isclose(unique, note, abs_tol=0.0001)):
            return index
    return None    

def predictNextNotes(input, steps, lstm_model, voices, scaler):
    # predicted notes
    predicted_notes = np.zeros((1,4))

    # all unique notes for each voice
    unique_voice1 = np.unique(voices[:,0])
    unique_voice2 = np.unique(voices[:,1])
    unique_voice3 = np.unique(voices[:,2])
    unique_voice4 = np.unique(voices[:,3])
    one_hot_values = np.concatenate((unique_voice1, unique_voice2, unique_voice3, unique_voice4))

    # BCEwithLogitLoss uses sigmoid when calculating loss, but we need to pass through
    sigmoid = nn.Sigmoid()

    # prepare input
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
    for i in range(steps):
        # print(input.shape)
        output = lstm_model(input, stateful=False)
        output = sigmoid(output)
        output = output.detach().numpy().squeeze()
        
        # get the indices with highest value from model forward output
        note_voice1 = np.argmax(output[:len(unique_voice1)])
        note_voice2 = np.argmax(output[len(unique_voice1) : len(unique_voice1) + len(unique_voice2)])
        note_voice3 = np.argmax(output[len(unique_voice1) + len(unique_voice2) : len(unique_voice1) + len(unique_voice2) + len(unique_voice3)])
        note_voice4 = np.argmax(output[-len(unique_voice4):])

        # get notes
        note_voice1 = one_hot_values[note_voice1]
        note_voice2 = one_hot_values[len(unique_voice1) + note_voice2]
        note_voice3 = one_hot_values[len(unique_voice1) + len(unique_voice2) + note_voice3]
        note_voice4 = one_hot_values[len(unique_voice1) + len(unique_voice2) + len(unique_voice3) + note_voice4]

        # add to array and inverse scale
        next_notes = np.array([note_voice1, note_voice2, note_voice3, note_voice4])
        next_notes_invscaled = scaler.inverse_transform(next_notes.reshape(1, -1))
        predicted_notes = np.concatenate((predicted_notes, next_notes_invscaled), axis = 0)

        # change input
        # drop oldest notes
        input = input[0][1:]
        # concat predicted notes
        input = torch.cat((input, torch.Tensor(next_notes).unsqueeze(0)))
        input = input.unsqueeze(0)

    return(predicted_notes.astype(np.int32)[1:])

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
    steps = 200
    new_music = predictNextNotes(input, steps, model, voices, scaler)

    # save new music
    np.savetxt(fname = "output/output.txt", X = new_music, fmt = "%d")

if __name__ == '__main__':
    torch.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(precision=3)
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=3)
    main()