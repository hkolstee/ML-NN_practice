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

def linear_deriv(x):
    return 0 # not sure if this should be like this

def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()

def softmax_deriv(x):
    x = softmax(x)
    jacob_matrix = np.diag(x)
    for i in range(len(jacob_matrix)):
        for j in range(len(jacob_matrix)):
            if (i == j):
                jacob_matrix[i][j] = x[i] * (1-x[i])
            else:
                jacob_matrix[i][j] = -x[i] * x[j]
    return jacob_matrix
    

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
                    activation:str, output_activation:str):
        self.input_size = input_size
        self.output_size = output_size
        self.nr_layers = nr_layers
        self.hidden_units = hidden_units

        # initialize the activations for use in backprop
        self.hidden_activ_vals = []
        for i, nr_neurons in enumerate(hidden_units):
            self.hidden_activ_vals.append(np.zeros(nr_neurons))

        # check if nr of layers is equal to nr of hidden_units sizes given in hidden_units array
        if (nr_layers != len(hidden_units)):
            raise IndexError("The number of layers ({}) specified is different then number of layers specified in the hidden_units array ({} = {}).".format(nr_layers, hidden_units, len(hidden_units)))
        
        # set activation function and its derivative
        self.__setActivFunc(activation)
        self.__setOutActivFunc(output_activation)

        # create topology of weights: input -> layers -> output
        w_topology = self.__createTopology(input_size, hidden_units, output_size)

        # initialize weights
        self.__initWeights(nr_layers, w_topology)

    # set activation functions and its derivative to call in the forward and backprop function
    def __setActivFunc(self, activ_func):
        # set activation function
        if (activ_func == "tanh"):
            self.activ_func = tanh
            self.activ_d_func = tanh_deriv
        elif (activ_func == "sigmoid"):
            self.activ_func = sigmoid
            self.activ_d_func = sigmoid_deriv
        elif (activ_func == "relu"):
            self.activ_func = relu
            self.activ_d_func = relu_deriv
        else:
            raise KeyError("Only activation functions available: \"tanh\", \"sigmoid\", \"relu\"")
        
    def __setOutActivFunc(self, out_activ_func):
        # set output activation function
        if (out_activ_func == "sigmoid"):
            self.out_activ_func = sigmoid
            self.out_activ_d_func = sigmoid_deriv
        elif (out_activ_func == "linear"):
            self.out_activ_func = linear
            self.out_activ_d_func = linear
        elif (out_activ_func == "softmax"):
            self.out_activ_func = softmax
            self.out_activ_d_func = softmax_deriv
        else:
            raise KeyError("Only activation functions available: \"linear\", \"sigmoid\", \"softmax\"")

    # create a list of the topology of units per layer: input -> hidden -> output
    #   we use this to initialize the weights
    def __createTopology(self, input_size, hidden_units, output_size):
        try:
            w_topology = [input_size] + hidden_units + [output_size]
        except TypeError:
            raise TypeError("hidden_units should be a list or array in the shape of (nr_layers,)") from None
        
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
            if (i == 0):
                self.hidden_activ_vals[i] = [self.activ_func(np.dot(out, node_weights)) for node_weights in weights]
            else:
                self.hidden_activ_vals[i] = [self.activ_func(np.dot(self.hidden_activ_vals[i-1], node_weights)) for node_weights in weights]
        
        # last layer to output
        out = [self.out_activ_func(np.dot(out, node_weights)) for node_weights in self.layers_weights[-1]]

        return out
    
    # backpropagation over network (SGD)
    def backprop(self, output, target, loss_function, lr):
        # forward pass
        # output = self.forward(input)


        # initialize gradients of weights
        gradients = []
        for i, weights in enumerate(self.layers_weights):
            gradients.append(np.zeros(weights.shape))
            print(weights.shape)

        # print(gradients[-1][0])

        # print(self.hidden_activ_vals[-1])
        # print(self.out_activ_d_func(output))
        # print("loss:", output, "to", target, "=", BCELoss_deriv(output, target))

        # output layer (either sigmoid, linear, or softmax)
        gradients[-1][0] = [BCELoss_deriv(output, target) * self.out_activ_d_func(output) * a for a in self.hidden_activ_vals[-1]]
        
        print(self.hidden_activ_vals)
        print(gradients)
        # print(self.layers_weights)

        for i in reversed(range(len(self.layers_weights) - 1)):
            for j in range(len(self.layers_weights[i])):
                gradients[i][j] = [gradients[i+1][j] * self.activ_d_func(self.hidden_activ_vals[i])] 

    # get weights 
    def weights(self):
        return self.layers_weights



nn = FeedForwardNN(input_size = 5, output_size = 1, 
                   nr_layers = 2, hidden_units = [5, 5], 
                   activation = "tanh", output_activation="sigmoid")
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
