## Regresja MSE
## Klasyfikacja binarna BCE
## Klasyfikacja wieloklasowa CCE

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


if __name__ == "__main__":
    X = [[1, 2, 3, 1], [1, 2, 3, 1], [1, 2, 3, 1]]  ## 3 na 4
    np.random.seed(0)
    layer1 = Layer_Dense(4, 5)  ## 4 i y liczba
    layer2 = Layer_Dense(5, 2)  ## y liczba i dowolna np 2
    ## wejsciowa 3 na 4 potem jest 3 na 5 a potem jest 3 na 2

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)


