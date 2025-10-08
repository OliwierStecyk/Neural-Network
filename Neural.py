## Regresja MSE
## Klasyfikacja binarna BCE
## Klasyfikacja wieloklasowa CCE

import math
import sys
import numpy as np
import matplotlib

## Usage
# simple mnożeniem przez stałą scale
# he  dla ReLU/PReLU
# xavier →uniform (Glorot)
# xavier_normal  normal (Glorot)
# random_normal normal 0–1
# random_uniform  uniform 0–1
# zero → wszystkie wagi zerowe (niezalecane)

class Layer_Dense: ## from https://www.geeksforgeeks.org/machine-learning/weight-initialization-techniques-for-deep-neural-networks/
    def __init__(self, n_inputs, n_neurons, init_type = "simple", scale = 0.2 ):
        self.output = 0
        init_type = init_type.lower()

        if init_type == "simple":
            self.weights = scale * np.random.randn(n_inputs, n_neurons )
        elif init_type == "he":
            self.weights = np.random.randn( n_inputs, n_neurons) * np.sqrt( 2, n_inputs )
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

### Activation Classes

class ActivationRelu: ## warstwy ukryte
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_PReLU: ## warstwy ukryte, gdy ReLU nie może
    def __init__(self, alpha = 0.02 ):
        self.alpha = alpha
    def forward(self, inputs ):
        self.output = np.where( inputs > 0, inputs, self.alpha * inputs )

class Activation_Softmax: ## for overflowing we are doing v = v - max(v) // wieloklasowe wyjście
    def forward(self, inputs ):
        exp_values =  np.exp( inputs - np.max( inputs, axis = 1, keepdims = True ) )
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        ## output == propabilities

class Activation_Sigmoid: ## zakres 0 - 1
    def forward(self, inputs):
        self.output = 1 / ( 1 + np.exp(-inputs) )

class Activation_tanh:
    def forward(self, inputs):
        self.output = np.tanh( inputs )

class Activation_Elu:
    def forward(self, inputs, alpha = 1.0 ):
        self.output = np.where( inputs >= 0, inputs, alpha * ( np.exp( inputs ) - 1 ))

class Activation_Elu:
    def forward(self, inputs, lambda_=1.0507, alpha=1.67326):
        self.output = lambda_ * np.where(inputs >= 0, inputs, alpha * (np.exp( inputs ) - 1))

class Activation_gelu:
    def forward(self, inputs):
        self.output =  0.5 * inputs * (1 + np.tanh(np.sqrt(2/np.pi) * ( inputs + 0.044715 * inputs**3 )))

### Losses Classes

class Loss:
    def calculate(self, output, y ):
        sample_losses = self.forward( output, y )
        data_losses = np.mean(sample_losses)
        return data_losses

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7 )

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
        negative_log_likelihoods = -np.log( correct_confidences )
        return negative_log_likelihoods

class Back: ## https://apxml.com/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/backpropagation-computing-gradients
    pass




if __name__ == "__main__":
    X = [[1, 2.9, 3, 3.5], [0, 2.6, -3.5, 1], [-1.8, -2, 3.1, 1.9]]

    ##E = math.e## 3 na 4
    ##exp_values = np.exp(X)
    ##E_LIST = [ [ E**x for x in row ] for row in X ]

    np.random.seed(0)

    layer1 = Layer_Dense(4, 5)  ## 4 i y liczba
    layer2 = Layer_Dense(5, 1)  ## y liczba i dowolna np 2
    ## wejsciowa 3 na 4 potem jest 3 na 5 a potem jest 3 na 2

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)


