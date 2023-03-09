import numpy as np

# activiation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return (1 - np.power(np.tanh(x), 2))

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_deriv(x):
    # prevent overflow
    x = np.clip(x, -500, 500)
    # derivative is
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
#   TODO: mean squared error
def MSELoss(output, target):
    pass

#   binary cross entropy
def BCELoss(output, target):
    return (-target * np.log(output) - (1 - target) * np.log(1 - output))

def BCELoss_deriv(output, target):
    return (-target / output) + ((1 - target) / (1 - output))


# simple feed forward network with multiple layers
# parameters:
#   input_size: The dimensionality of an input sample
#   output_size: The dimensionality of the network output
#   nr_layers: Number of hidden layers
#   hidden_units: The size of the hidden layers (a list of sizes, one size for each hidden layer)
#   activation: The activation function used in the hidden layers
#   output_activation: The acctivation used in the last hidden to output layer
class FeedForwardNN:
    def __init__(self, input_size:int, output_size:int, 
                    nr_layers:int, hidden_units:list, 
                    activation:str, output_activation:str):
        self.input_size = input_size
        self.output_size = output_size
        self.nr_layers = nr_layers
        self.hidden_units = hidden_units

        # initialize the weighted sums and activations for use in backprop
        self.activ_vals = []
        self.weighted_sums = []
        for i, nr_neurons in enumerate(hidden_units):
            self.activ_vals.append(np.zeros(nr_neurons))
            self.weighted_sums.append(np.zeros(nr_neurons))

        # check if nr of layers is equal to nr of hidden_units sizes given in hidden_units array
        if (nr_layers != len(hidden_units)):
            raise IndexError("The number of layers ({}) specified is different then number of layers specified in the hidden_units array ({} = {}).".format(nr_layers, hidden_units, len(hidden_units)))
        
        # set activation function and its derivative
        self.__setActivFunc(activation)
        self.__setOutActivFunc(output_activation)

        # create topology of weights: input -> layers -> output
        w_topology = self.__createTopology(input_size, hidden_units, output_size)
        # print("wtopo:", w_topology)

        # initialize weights
        self.__initWeights(nr_layers, w_topology)
        
        # set gradients to 0
        self.__resetGradients()

    # set activation functions and its derivative to call in the forward and backprop function
    def __setActivFunc(self, activ_func):
        # set activation function
        if (activ_func == "tanh"):
            self.activ_func = tanh
            self.activ_func_deriv = tanh_deriv
        elif (activ_func == "sigmoid"):
            self.activ_func = sigmoid
            self.activ_func_deriv = sigmoid_deriv
        elif (activ_func == "relu"):
            self.activ_func = relu
            self.activ_func_deriv = relu_deriv
        else:
            raise KeyError("Only activation functions available: \"tanh\", \"sigmoid\", \"relu\"")
        
    def __setOutActivFunc(self, out_activ_func):
        # set output activation function
        if (out_activ_func == "sigmoid"):
            self.out_activ_func = sigmoid
            self.out_activ_func_deriv = sigmoid_deriv
        elif (out_activ_func == "linear"):
            self.out_activ_func = linear
            self.out_activ_func_deriv = linear
        elif (out_activ_func == "softmax"):
            self.out_activ_func = softmax
            self.out_activ_func_deriv = softmax_deriv
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

    # reset gradients to 0
    def resetGradients(self):
        self.gradients = []
        for i, weights in enumerate(self.layers_weights):
            self.gradients.append(np.zeros(weights.shape))

    # forward through the model
    def forward(self, input):
        # check input size
        if (len(input) != self.input_size):
            raise IndexError("Size of input ({}) and network input size ({}) are not equal.".format(len(input), self.input_size))
        
        # forward through the network (activation func on the weighted sum for each neuron in layer).
        #   The weighted sums and activation values are saved in list of matrices to be used in backprop.
        #   TODO: Can be implemented to only do this when training network.
        out = input
        for i, weights in enumerate(self.layers_weights[:-1]):
            if (i == 0):
                self.weighted_sums[i] = np.array([np.dot(out, node_weights) for node_weights in weights])
                self.activ_vals[i] = np.vectorize(self.activ_func)(self.weighted_sums[i])
            else:
                self.weighted_sums[i] = np.array([self.activ_func(np.dot(self.activ_vals[i-1], node_weights)) for node_weights in weights])
                self.activ_vals[i] = np.vectorize(self.activ_func)(self.weighted_sums[i])

        # last layer to output
        out = [self.out_activ_func(np.dot(self.activ_vals[-1], node_weights)) for node_weights in self.layers_weights[-1]]

        return out
    
    # backpropagation over network (SGD)
    def backprop(self, input, output, target, loss_function):
        # check loss function
        if (loss_function == "BCELoss"):
            loss_func = BCELoss
            loss_func_deriv = BCELoss_deriv
        else:
            raise KeyError("Only loss functions available: \"BCELoss\", \"MSELoss\"")

        # first output layer gradients (either sigmoid, linear, or softmax).
        #   calculates: gradient = dLoss/dWeight = dLoss/dActiv_value * dActiv_value/dWsum * dWsum/dWeight
        #       as can be concluded from chainrule
        #       where: delta = dLoss/dActiv_value * dActiv_value/dWsum
        #           for use in gradient calculation of deeper layers
        # more info on inner workings: towardsdatascience.com -> understanding backpropagation
        #   np.vectorize(function)(array) works as map(function, array) function
        delta = loss_func_deriv(output, target) * np.vectorize(self.out_activ_func_deriv)(self.weighted_sums[-1])
        self.gradients[-1] += delta * self.activ_vals[-1]    
        # rest of network weight gradients 
        # this is calculated by: gradient = dot(delta^T, dWsum/dActiv_value) * dActiv_value/dWsum * dWsum/dWeights
        for i in reversed(range(len(self.gradients) - 1)):
            # reached end -> need to use input as activation values
            if (i == 0):
                delta = np.dot(delta, np.squeeze(self.layers_weights[i+1])) * np.vectorize(self.activ_func_deriv)(self.weighted_sums[i])
                self.gradients[i] += np.array([input * error for error in delta])
            # somewhere within hidden layer weights
            else:
                delta = np.dot(delta, np.squeeze(self.layers_weights[i+1])) * np.vectorize(self.activ_func_deriv)(self.weighted_sums[i])
                self.gradients[i] += np.array([self.activ_vals[i-1] * error for error in delta])

    # take a step along the negative build up gradient (can be batch, or single sample, whenever you like)
    #   backpropage over any number of samples, the build up gradient gets stored in the network
    #   call step(), and a step is taken down this gradient, and the gradient is reset.
    def step(self, lr):
        # change weights based on negative gradient times learning rate (SGD)
        for i in range(len(self.layers_weights)):
            self.layers_weights[i] = self.layers_weights[i] - lr * self.gradients[i]

        # reset gradients
        self.__resetGradients()

    # get weights 
    def weights(self):
        return self.layers_weights
