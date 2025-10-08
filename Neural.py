## Regresja MSE
## Klasyfikacja binarna BCE
## Klasyfikacja wieloklasowa CCE

import math
import sys
import numpy as np
import matplotlib

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons ):
        self.output = 0
        self.weights = 0.20 * np.random.randn(n_inputs, n_neurons )
        self.biases = np.zeros( ( 1, n_neurons ))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationRelu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_PReLU:
    def __init__(self, alpha = 0.01 ):
        self.alpha = alpha
    def forward(self, inputs ):
        self.output = np.where( inputs > 0, inputs, self.alpha * inputs )

class Activation_Softmax: ## for overflowing we are doing v = v - max(v)
    def forward(self, inputs ):
        exp_values =  np.exp( inputs - np.max( inputs, axis = 1, keepdims = True ) )
        self.output = np.exp(exp_values)/ sum( np.exp(exp_values, axis = 1, keepdims = True))
        ## output == propabilities

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


