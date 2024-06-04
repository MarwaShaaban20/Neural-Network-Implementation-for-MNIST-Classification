
# Neural Network Implementation for MNIST Classification



This project implements a simple neural network from scratch to classify handwritten digits from the MNIST dataset. The code includes steps for data preprocessing, network training, and evaluation.

### Features

- **Data Preprocessing:**
  - Standardization of training and test data.
  - Reshaping the images to a flat vector.
  - Splitting the training data into training and validation sets.
  - One-hot encoding of the labels.

- **Neural Network Implementation:**
  - Sigmoid activation function and its derivative.
  - Mean Squared Error (MSE) loss function.
  - Weight initialization.
  - Forward and backward propagation.
  - Gradient descent for weight updates.

- **Training and Evaluation:**
  - Training the network using mini-batch gradient descent.
  - Monitoring training and validation loss.
  - Evaluating the model on test data.
  - Printing the accuracy and loss on the test set.

### Code Overview

1. **Data Loading and Preprocessing:**
    - Load MNIST dataset.
    - Standardize the images.
    - Reshape images to 1D vectors.
    - Split the data into training and validation sets.
    - One-hot encode the labels.

2. **Neural Network Functions:**
    - `sigmoid`: Computes the sigmoid activation.
    - `sigmoid_derivative`: Computes the derivative of the sigmoid function.
    - `mse_loss`: Computes the Mean Squared Error loss.
    - `initialize_weights`: Initializes weights and biases.
    - `softmax`: Computes the softmax for output probabilities.
    - `forward`: Implements the forward pass.
    - `backward`: Implements the backward pass using backpropagation.
    - `update_weights`: Updates the weights and biases using gradient descent.

3. **Training Function:**
    - `train`: Trains the neural network using the specified architecture and hyperparameters.
    - Prints training and validation loss for each epoch.

4. **Evaluation Function:**
    - `NN`: Initializes, trains, and evaluates the neural network on the test data.
    - Prints the test accuracy and loss.
    - Compares true labels with predicted labels.

5. **Architecture Testing:**
    - Tests different neural network architectures with varying layers and neurons.
    - Prints accuracy for each architecture.



### Example Architectures

- **Architecture 1:**
  - Layers: [784, 20, 10]
- **Architecture 2:**
  - Layers: [784, 20, 15, 10]
- **Architecture 3:**
  - Layers: [784, 15, 20, 10]

Each architecture is tested and evaluated, and the results are printed to the console.

### Results

The script will output the accuracy and loss for each architecture on the test set, along with the predicted and true labels for comparison.

---


