import numpy as np

# activiation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return (1 - np.power(np.tanh(x), 2))

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_deriv(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

def relu(x):
    return max(0., x)

def relu_deriv(x):
    return (1 if x > 0 else 0)

def linear(x):
    return x

# loss functions
#   mean squared error
def MSELoss(output, target):
    pass

#   binary cross entropy
def BCELoss(output, target):
    return (-target * np.log(output) - (1 - target) * np.log(1 - output))

def BCELoss_deriv(output, target):
    return (-target / output) + ((1 - target) / (1 - output))


# simple feed forward network with multiple layers given with nr_layers, hidden_units
#                                                       where nr_layers = len(hidden_units)
# last layer -> output is linear activation
class FeedForwardNN:
    def __init__(self, input_size:int, output_size:int, 
                    nr_layers:int, hidden_units:list, 
                    activation:str):
        self.input_size = input_size
        self.output_size = output_size
        self.nr_layers = nr_layers
        self.hidden_units = hidden_units

        # check if nr of layers is equal to nr of hidden_units sizes given in hidden_units array
        if (nr_layers != len(hidden_units)):
            raise IndexError("The number of layers ({}) specified is different then number of layers specified in the hidden_units array ({} = {}).".format(nr_layers, hidden_units, len(hidden_units)))
        
        # set activation function and its derivative
        self.__setActivation(activation)

        # create topology of weights: input -> layers -> output
        w_topology = self.__createTopology(input_size, hidden_units, output_size)

        # initialize weights
        self.__initWeights(nr_layers, w_topology)

    # set activation functions and its derivative to call in the forward and backprop function
    def __setActivation(self, activation):
        # set activation function
        if (activation == "tanh"):
            self.activ = tanh
            self.activ_deriv = tanh_deriv
        elif (activation == "sigmoid"):
            self.activ = sigmoid
            self.activ_deriv = sigmoid_deriv
        elif (activation == "relu"):
            self.activ = relu
            self.activ_deriv = relu_deriv
        else:
            raise KeyError("Only activation functions available: \"tanh\", \"sigmoid\", \"relu\"")

    # create a list of the topology of units per layer: input -> hidden -> output
    #   we use this to initialize the weights
    def __createTopology(self, input_size, hidden_units, output_size):
        try:
            w_topology = [input_size] + hidden_units + [output_size]
        except TypeError:
            raise TypeError("hidden_units should be a list or array in the size of nr_layers") from None
        
        return w_topology

    # initialize the weights of the entire model
    def __initWeights(self, nr_layers, w_topology):
        self.layers_weights = []

        # loop through hidden_units array -> initialize weights to next hidden_units layer
        for i in range(nr_layers + 1):
            # create weights for current hidden_units layer to the next (xavier initialization)
            current_to_next = np.random.uniform(low = -(1/np.sqrt(w_topology[i])), 
                                                high = (1/np.sqrt(w_topology[i])), 
                                                size = (w_topology[i+1], w_topology[i]))

            # append to list
            self.layers_weights.append(current_to_next)

    # forward through the model
    def forward(self, input):
        # check input size
        if (len(input) != self.input_size):
            raise IndexError("Size of input ({}) and network input size ({}) are not equal.".format(len(input), self.input_size))
        
        # forward through the network (activation func on the weighted sum for each neuron in layer)
        out = input
        for i, weights in enumerate(self.layers_weights[:-1]):
            out = [self.activ(np.dot(out, node_weights)) for node_weights in weights]

        # last layer to output is linear
        # out = [np.dot(out, node_weights) for node_weights in self.layers_weights[-1]]

        # last layer to output is sigmoid
        out = [sigmoid(np.dot(out, node_weights)) for node_weights in self.layers_weights[-1]]

        return out
    
    # backpropagation over network (SGD)
    def backprop(self, output, target, loss_function, lr):
        # forward pass
        # output = self.forward(input)

        # compute loss
        loss = BCELoss(output, target)

        print("Loss:", output, target, " = ", loss)

        
    # get weights 
    def weights(self):
        return self.layers_weights



nn = FeedForwardNN(input_size = 5, output_size = 1, 
                   nr_layers = 3, hidden_units = [12, 8, 6], 
                   activation = "tanh")
weights = nn.weights()
print(len(weights))
for i in range(len(weights)):
    print(weights[i].shape)
    print(weights[i])

input = np.random.rand(5)
print("\ninput", input)
output = nn.forward(input)
print("output", output)
nn.backprop(output[0], 1, "loss funct here i guess", 0.01)
