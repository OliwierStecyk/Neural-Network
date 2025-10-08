# **Train Your First Neural Network from Scratch**

This repository contains a step-by-step implementation of a neural network built entirely from scratch in Python using only NumPy. It is designed for educational purposes to demonstrate how neural networks work “under the hood” — including forward propagation, backpropagation, gradient descent, and weight updates.

---

## **Features**

* Implements a simple neural network that can learn XOR or Iris dataset classification.
* Visualizes training progress: loss over time and decision boundaries.
* Allows experimenting with different architectures, activation functions, and learning rates.
* Fully written in Python with NumPy — no high-level ML frameworks required.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/Train-Your-First-Neural-Network.git
cd Train-Your-First-Neural-Network
```

2. Install dependencies:

```bash
pip install numpy matplotlib
```

---

## **Usage**

Run the main script to train the network and see the results:

```bash
python train_xor.py
```

* For Iris dataset:

```bash
python train_iris.py
```

* Training progress and decision boundaries will be displayed as plots.

---

## **Project Structure**

```
/Train-Your-First-Neural-Network
│
├─ train_xor.py          # XOR example
├─ train_iris.py         # Iris dataset example
├─ neural_network.py     # Core network code (forward/backprop)
├─ utils.py              # Helper functions (visualization, data processing)
└─ README.md
```

---

## **Example Output**

* Loss decreasing over epochs
* Decision boundary changing during training
* Final predictions for the dataset

*(You can include GIFs or images of training progress here)*

---

## **Learning Goals**

* Understand forward propagation and backpropagation in neural networks.
* Learn how gradient descent updates weights.
* Explore the effect of network architecture, activation functions, and learning rate on training.

---


---

## **Activation Functions**

This project supports a variety of activation functions for hidden and output layers. Activation functions allow the network to model non-linear relationships in the data.

### **Hidden Layer Activations**

Hidden layers are the intermediate layers between the input and output. Their main purpose is to **learn complex, non-linear patterns** in the data.

| Activation | Output Range | Description                                     | When to Use                                             |
| ---------- | ------------ | ----------------------------------------------- | ------------------------------------------------------- |
| **ReLU**   | `[0, ∞)`     | Rectified Linear Unit: `max(0, x)`              | Most common hidden layer activation                     |
| **PReLU**  | `(-∞, ∞)`    | Parametric ReLU: `x` if x>0 else `alpha*x`      | Helps when ReLU neurons “die” (always 0)                |
| **tanh**   | `[-1, 1]`    | Hyperbolic tangent                              | Good for symmetric data in hidden layers                |
| **ELU**    | `(-∞, ∞)`    | Exponential Linear Unit: smooth negative values | Can improve learning compared to ReLU                   |
| **GELU**   | `(-∞, ∞)`    | Gaussian Error Linear Unit: smooth, non-linear  | Often used in modern architectures (e.g., Transformers) |

**Example:**

```python
layer_hidden = Layer_Dense(n_inputs=4, n_neurons=5)
activation_hidden = ActivationReLU()
activation_hidden.forward(layer_hidden.output)
```

---

### **Output Layer Activations**

The output layer activation is chosen depending on the **type of problem** you are solving. It ensures that the network output is in a suitable form for the task:

| Activation        | Output Range       | Use Case                                                  |
| ----------------- | ------------------ | --------------------------------------------------------- |
| **Sigmoid**       | `[0, 1]`           | Binary classification (output interpreted as probability) |
| **Softmax**       | `[0, 1]` (sum = 1) | Multi-class classification (probabilities for each class) |
| **None / Linear** | `(-∞, ∞)`          | Regression tasks                                          |

**Example:**

```python
layer_output = Layer_Dense(n_inputs=5, n_neurons=3)
activation_output = Activation_Softmax()
activation_output.forward(layer_output.output)
```

---

### **Key Differences**

| Feature           | Hidden Layer                                     | Output Layer                                    |
| ----------------- | ------------------------------------------------ | ----------------------------------------------- |
| Purpose           | Learn complex patterns in the data               | Produce final prediction for the task           |
| Activation choice | Usually non-linear: ReLU, PReLU, tanh, ELU, GELU | Task-specific: Sigmoid, Softmax, or Linear      |
| Output            | Can have arbitrary range                         | Must match the problem’s expected output        |
| Example           | `ActivationReLU()`                               | `ActivationSigmoid()` for binary classification |

---


## **License**

This project is open-source and available under the MIT License.

---

