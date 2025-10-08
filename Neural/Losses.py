import numpy as np

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

class Loss_MSE(Loss):
    def forward(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2, axis=-1)

class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        return -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
