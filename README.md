# Deep Neural Network Implementation

This was a project for my machine learning class where were tasked in implementing a feedforward neural network with an arbitrary number of hidden layers, configurable activation functions, and support for both regression and classification tasks. The network is trained using backpropagation and gradient descent optimization. This was made <ins>only</ins> using numpy and other built in python packages in order to further understand DNN's at a higher level.

## Key Features

- **Configurable Network Architecture**: The number of hidden layers, number of units in each hidden layer, and the activation function used for hidden layers can be specified as command-line arguments.

- **Regression and Classification**: The code supports both regression and classification tasks. For classification, it supports multi-class classification with softmax output activation and cross-entropy loss.

- **Data Loading**: The code can load training and validation data from CSV files, where features and targets are separated into different files.

- **Minibatch Training**: Training can be performed with minibatches of data, with the option for full-batch training by setting the minibatch size to 0.

- **Verbose Mode**: A verbose mode can be enabled to display training and validation loss/accuracy after each update step during training.

- **Weight Initialization**: Weights are initialized randomly within a specified range.

## Usage

The code can be executed from the command line with various arguments to configure the network architecture and training process. The available arguments are:

- `-v`: Enable verbose mode.
- `-train_feat`: Path to the training feature file.
- `-train_target`: Path to the training target file.
- `-dev_feat`: Path to the development (validation) feature file.
- `-dev_target`: Path to the development (validation) target file.
- `-epochs`: Number of epochs to train (default: 100).
- `-learnrate`: Learning rate (default: 0.01).
- `-nunits`: Number of hidden units (default: 10).
- `-type`: Problem type, 'R' for regression or 'C' for classification (default: 'R').
- `-hidden_act`: Activation function for hidden units, options: 'tanh', 'relu', 'sigmoid' (default: 'relu').
- `-init_range`: Initialization range for weights (default: 0.1).
- `-num_classes`: Number of classes for classification (default: 2).
- `-mb`: Minibatch size (default: 1).
- `-nlayers`: Number of hidden layers (default: 1).

