import numpy as np

def FeedForwardNN():
    def __init__(self, input_size:int, output_size:int, layers:int, hidden:np.ndarray):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.hidden = hidden
        
        # first initialize between hidden weights
        try:
            self.hidden_w = np.ndarray([])

            # loop through hidden array -> initialize weights to next hidden layer
            for i in range(layers - 1):
                # create weights for current hidden layer to the next
                current_to_next = np.zeros((hidden[i], hidden[i+1]), dtype=np.float32)                
                self.hidden_w = np.append(self.hidden_w, (current_to_next))