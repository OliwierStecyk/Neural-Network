import numpy as np

class ActivationRelu: ## warstwy ukryte

    def forward(self, inputs ):
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues ):
        self.dinputs = dvalues.copy()
        self.dinputs[ self.inputs <= 0 ] = 0

class Activation_PReLU: ## warstwy ukryte, gdy ReLU nie może

    def __init__(self, alpha = 0.02 ):
        self.alpha = alpha

    def forward(self, inputs ):
        self.output = np.where( inputs > 0, inputs, self.alpha * inputs )

    def bacward(self, dvalues ):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] *= self.alpha

class Activation_Softmax: ## for overflowing we are doing v = v - max(v) // wieloklasowe wyjście

    def forward(self, inputs ):
        exp_values =  np.exp( inputs - np.max( inputs, axis = 1, keepdims = True ) )
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        ## output == propabilities
    def bacward( self, dvalues ):

        self.dinputs = np.empty_like(dvalues)

        for index, ( single_output, single_dvalues ) in enumerate( zip ( self.output, dvalues )):
            single_output = single_output.reshape( -1, 1 )
            jacobian_matrix = np.diagoflat( single_output )

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


