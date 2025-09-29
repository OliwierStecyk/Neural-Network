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

## **License**

This project is open-source and available under the MIT License.

---

