import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, init_type = "simple", scale = 0.2 ):
        self.output = 0
        init_type = init_type.lower()

        if init_type == "simple":
            self.weights = scale * np.random.randn(n_inputs, n_neurons )
        elif init_type == "he":
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        elif init_type == "xavier":
            limit = np.sqrt(6 / (n_inputs + n_neurons))
            self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        elif init_type == "xavier_noraml":
            stddev = np.sqrt(2 / (n_inputs + n_neurons))
            self.weights = np.random.randn(n_inputs, n_neurons) * stddev
        elif init_type == "random_normal":
            self.weights = np.random.randn(n_inputs, n_neurons)
        elif init_type == "random_uniform":
            self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        elif init_type == "zero":
            self.weights = np.zeros(( n_inputs, n_neurons ))
        else:
            raise ValueError(f"Nieznany init_type: {init_type}")

        self.biases = np.zeros( ( 1, n_neurons ))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues ):
        self.dweights = np.dot( self.inputs.T, dvalues )
        self.dbiases = np.sum( dvalues, axis = 0, keepdims = True )
        self.dinputs = np.dot( dvalues, self.weights.T )