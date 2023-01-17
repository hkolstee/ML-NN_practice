import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim

import sys

# Define function to create train test data
def train_test(voice):

    train = []
    test = []
    
    # initialize how far in the past you want to look
    memory_window = 20
    
    # get a list of all the unique notes in the voice
    voice_set = set(voice)
    distribution_notes = list(voice_set)

    for i in range(len(voice) - (memory_window+1)):
        # the train data is a time-window of voices
        train.append(voice[i:i+ memory_window])
        
        # the test data is a probability one-hot encoded vector
        # create empty probability vector
        probability_vector = [0] * len(distribution_notes)
        # get the next note and find its index in the distribution
        next_note = voice[i+ memory_window + 1]
        idx = distribution_notes.index(next_note)
        # change the value at that position to 1 in the empty probability vector
        # and add it to the test set
        probability_vector[idx] = 1
        test.append(probability_vector)

    train = np.array(train)
    test = np.array(test)
    
    return train, test

def cal_offset(voice):
    
    # get the max and min values of a voice
    voice_max = np.amax(voice)
    # sort the array and keep only the unique values, because '0' is not a chord but a break,
    # so it shouldnt be the 'min value'.
    voice_sort = np.sort(voice)
    voice_sort = np.unique(voice_sort)
    voice_min = voice_sort[1]
    
    # calculate voice offset as: 2log2(voice_max) - 2log2(voice_min)/2 - 2log2(voice_max)
    voice_offset = (2 * math.log2(voice_max) - 2 * math.log2(voice_min))/2
    voice_offset = voice_offset - (2* math.log2(voice_max))
    
    return voice_offset

def chroma_circle(note):
    theta = (2*math.pi*(note % 12)/12)
    
    # r is the scaling set to 1
    r = 1   
    x = math.cos(theta) * r
    y = math.sin(theta) * r
    
    return x, y

def fifth_circle(note):
    theta = (2*math.pi*(7*note % 12)/12)
    
    # r is the scaling set to 1
    r = 1   
    x = math.cos(theta) * r
    y = math.sin(theta) * r
    
    return x, y

def voice_representation(voice, data):
    
    # list to save values
    new_representation = []
    
    # calculate offset of voice
    offset = cal_offset(voice)
    
    try:
        for training_data in data:
            # vector to save the new representation
            vector = []
            for note in training_data:
                # if the note is 0, it's a break and should be noted as [0 0 0 0 0] ??
                if note == 0:
                    vector.append(0)
                    vector.append(0)
                    vector.append(0)
                    vector.append(0)
                    vector.append(0)
                    continue

                # normalized frequency = 2log2(note)+offset(voice)
                voice_norm = 2*math.log2(note) + offset

                x_chroma, y_chroma = chroma_circle(note)
                x_fifth, y_fifth = fifth_circle(note)

                vector.append(voice_norm)
                vector.append(x_chroma)
                vector.append(y_chroma)
                vector.append(x_fifth)
                vector.append(y_fifth)
    except:
            # vector to save the new representation
            vector = []
            for note in data:
                # if the note is 0, it's a break and should be noted as [0 0 0 0 0] ??
                if note == 0:
                    vector.append(0)
                    vector.append(0)
                    vector.append(0)
                    vector.append(0)
                    vector.append(0)
                    continue

                # normalized frequency = 2log2(note)+offset(voice)
                voice_norm = 2*math.log2(note) + offset

                x_chroma, y_chroma = chroma_circle(note)
                x_fifth, y_fifth = fifth_circle(note)

                vector.append(voice_norm)
                vector.append(x_chroma)
                vector.append(y_chroma)
                vector.append(x_fifth)
                vector.append(y_fifth)
    new_representation.append(vector)
        
    return np.array(new_representation)


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

        print("LSTM initialized with {} input size, {} hidden layer size, {} number of LSTM layers, and an output size of {}".format(input_size, hidden_size, num_layers, output_size))
    
    def forward(self, input):
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size) 
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)
        
        out, (hn, cn) = self.lstm(input, (h0, c0)) 
        out = self.linear(out)  
        return out

# Define the training loop
def training(LSTM_model, n_epochs, optimizer, criterion, train, test):
    for i in range(n_epochs):
        optimizer.zero_grad()
        predictions = LSTM_model(train)
        # squeeze prediction because of broadcasting error (have to test if this even helps)
        loss = criterion(predictions, test)
        loss.backward()

        if (i % 10 == 0):
            print("Epoch:", i, "  Loss:", loss.item())
     
        optimizer.step()

    return (predictions.detach().numpy())

def main():    
    # Load voices
    voices = np.loadtxt("F.txt", dtype=np.int8)

    # Delete starting silence
    voices = np.delete(voices, slice(8), axis=0)

    # Split voices
    voice_one = voices[:,0]
    voice_two = voices[:,1]
    voice_three = voices[:,2]
    voice_four = voices[:,3]

    # Check the amount of unique notes in the voices
    # This is important for the model input dimensions when creating the model
    voice_one_unique = set(voice_one)
    voice_two_unique = set(voice_two)
    voice_three_unique = set(voice_three)
    voice_four_unique = set(voice_four)
    print("{} unique notes are found in voice one".format(len(voice_one_unique)))
    print("{} unique notes are found in voice two".format(len(voice_two_unique)))
    print("{} unique notes are found in voice three".format(len(voice_three_unique)))
    print("{} unique notes are found in voice four".format(len(voice_four_unique)))
    
    # Create train and test data
    train, test = train_test(voice_one)

    # Test different representation
    # voice_one_train_rep = voice_representation(voice_one, train)
    # print(voice_one_train_rep)
    
    # Define the model
    input_size = np.shape(train)[1]
    hidden_size = 32
    num_layers = 1
    output_size = np.shape(test)[1]
    LSTM_bach_model = LSTM_model(input_size, hidden_size, num_layers, output_size)

    # Define training parameters
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_bach_model.parameters(), lr=0.08)

    train_tensor = torch.FloatTensor(train).unsqueeze(0)
    test_tensor = torch.FloatTensor(test).unsqueeze(0)

    epochs = 100

    training(LSTM_bach_model, 100, optimizer, criterion, train_tensor, test_tensor)


if __name__ == "__main__":
    main()