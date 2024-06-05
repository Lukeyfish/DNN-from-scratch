import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# main method to load data and start training script
def main(args):
    train_feat, train_target, val_feat, val_target = load_data(args) # loads data

    train_set = Dataloader(train_feat, train_target, args.mb) # train set
    val_set = Dataloader(val_feat, val_target, args.mb) # val set

    weights, biases = init_weights(args, train_set.feature_len) # initializes weights and biases
    train(args, train_set, val_set)

# loads the data into memory
def load_data(args):
    train_feat = np.array(pd.read_csv(args.train_feat, header=None, delimiter=' ').values)
    train_target = np.array(pd.read_csv(args.train_target, header=None, delimiter=' ').values)
    val_feat = np.array(pd.read_csv(args.dev_feat, header=None, delimiter=' ').values)
    val_target = np.array(pd.read_csv(args.dev_target, header=None, delimiter=' ').values)
    return train_feat, train_target, val_feat, val_target


# softmax outputs
def softmax(X):
    exp_x = np.exp(X - np.max(X, axis = 0, keepdims = True)) # applies only across the mb axis instead of applying to everything at once
    return exp_x / np.sum(exp_x, axis = 0, keepdims = True)

# ReLU activation function
def relu(X):
    return np.maximum(0, X)

# ReLU derivative activation function
def relu_d(X):
    return np.where(X > 0, 1, 0)

# sigmoid activation function
def sigmoid(X):
    return (1 / (1 + np.exp(-X)))
    
# sigmoid derivative activation function
def sigmoid_d(X):
    return sigmoid(X) * (1 - sigmoid(X))

# tanh activation function
def tanh(X):
    return np.tanh(X)

# tanh derivative activation function
def tanh_d(X):
    return 1 - np.tanh(X) ** 2

# mse loss
def mse(y, yhat):
    return(.5 * (np.sum((np.square(y-yhat))))/len(yhat))

# mse derivative 
def mse_d(y, yhat):
    return yhat - y

# cross entropy loss
def cross_entropy(y, yhat):
    eps = 1e-10
    return -np.sum(y * np.log(yhat + eps))

# cross entropy loss derivative
def cross_entropy_d(y, yhat):
    return yhat - y

# gives the accuracy metric for predicted and true labels
def accuracy(y, yhat):
    y_pred = np.argmax(yhat, axis = 0)
    y_true = np.argmax(y, axis = 0)
    M = confusion_matrix(y_true, y_pred, labels = list(range(10))) # confusion matrix for testing
    num_correct = np.sum(y_pred == y_true)
    return num_correct, y_true.size, num_correct/(y_true.size), M

# finds the loss fore regression or classifcation given y and yhat
def compute_loss(args, y, yhat):
    if(args.type=='C'):
        return cross_entropy(y, yhat)
    elif(args.type=='R'):
        return mse(y, yhat)


# creates one hot representations
def one_hot(outputs, C):
    one_hot = np.zeros((C,outputs.shape[1]))
    one_hot[outputs, np.arange(outputs.shape[1])] = 1
    return one_hot

# initializes weights and biases based on input, hidden, and output sizes
def init_weights(args, input_size):
    weights = []
    biases = []

    i2h_w = np.random.uniform(-args.init_range, args.init_range, (args.nunits, input_size))
    i2h_b = np.random.uniform(-args.init_range, args.init_range, (args.nunits, 1))
    weights.append(i2h_w)
    biases.append(i2h_b)

    for hidden_layer in range(1, args.nlayers):
        h2h_w = np.random.uniform(-args.init_range, args.init_range, (args.nunits, args.nunits))
        h2h_b = np.random.uniform(-args.init_range, args.init_range, (args.nunits, 1))
        weights.append(h2h_w)
        biases.append(h2h_b)

    h2o_w = np.random.uniform(-args.init_range, args.init_range, (args.num_classes, args.nunits))
    h2o_b = np.random.uniform(-args.init_range, args.init_range, (args.num_classes, 1))
    weights.append(h2o_w)
    biases.append(h2o_b)

    return weights, biases

# does forward pass of network with nlayers
def forward(args, x, weights, biases):
    A = globals()[args.hidden_act] # sets the activation function inbetween layers

    As = [] #activation output for calculating gradients later
    Zs =[] #outputs for calculating gradients later

    As.append(x)
    Z_1 = weights[0] @ x + biases[0] 
    Zs.append(Z_1)
    A_1 = A(Z_1) 
    As.append(A_1)

    A_x = A_1
    for i in range(1, args.nlayers):
        Z_x = weights[i] @ A_x + biases[i]
        Zs.append(Z_x)
        A_x = A(Z_x)
        As.append(A_x)

    Z_out = weights[-1] @ A_x + biases[-1]
    Zs.append(Z_out)

    if(args.num_classes > 2): # if doing multi-class classification, output activation = softmax
        Z_out = softmax(Z_out)
    elif(args.num_classes == 2): # if doing multi-class classification, output activation = sigmoid
        Z_out = sigmoid(Z_out)

    return Zs, As, Z_out

# backprop algorithm, returns the grads to be used for updating
def back_prop(args, Zs, As, weights, biases, x, y, yhat):
    A = globals()[(args.hidden_act + '_d')] # sets the activation function inbetween layers
    
    weights_g = []
    biases_g = []

    if(args.type == 'C'):
        layer_g = cross_entropy_d(y, yhat)
    elif(args.type == 'R'):
        layer_g = mse_d(y, yhat)

    for i in reversed(range(len(weights))):
        if i != len(weights)-1:
            layer_g = np.multiply(layer_g, relu_d(Zs[i]))

        layer_w = layer_g @ As[i].T
        layer_b = np.mean(layer_g, axis=1, keepdims=True)
        weights_g.insert(0, layer_w)
        biases_g.insert(0, layer_b)

        layer_g = weights[i].T @ layer_g
    return weights_g, biases_g

# given the gradients, updateds weights & biases accordingly
def update_weights(args, weights_g, biases_g, weights, biases):
    for i in range(0,len(weights)):
        weights[i] -= weights_g[i] * args.learnrate
        biases[i] -= biases_g[i] * args.learnrate
    return weights, biases


# training loop
def train(args, train_set, val_set):
    weights, biases = init_weights(args, train_set.feature_len)
    for epoch in range(0, args.epochs):
        train_set._shuffle_data()
        val_set._shuffle_data()

        toss = 0
        voss = 0

        train_num_correct = 0
        train_total_number = 0

        for i in range(train_set.num_batches()):
            x, y = train_set[i]
            if(args.type=='C'): 
                y = one_hot(y, args.num_classes)
            Zs, As, yhat = forward(args, x, weights, biases)

            if(args.type=='C'):
                num_correct, total, batch_acc, M = accuracy(y, yhat)
                train_num_correct += num_correct
                train_total_number += total
            
            toss += compute_loss(args, y, yhat)
            weights_g, biases_g = back_prop(args, Zs, As, weights, biases, x, y, yhat)
            weights, biases = update_weights(args, weights_g, biases_g, weights, biases)

            # resets values for accuracy calculation
            val_num_correct = 0
            val_total_number = 0

            if(args.v == True): # if in verbose mode, runs through val once each update step
                for j in range(val_set.num_batches()):
                    x, y = val_set[j]
                    if(args.type == 'C'):
                        y = one_hot(y, args.num_classes)
                    Zs, As, yhat = forward(args, x, weights, biases)

                    if(args.type == 'C'):
                        num_correct, total, batch_acc, M = accuracy(y, yhat)
                        val_num_correct += num_correct
                        val_total_number += total
                    voss += compute_loss(args, y, yhat)

                if(args.type == 'C'):
                    toss = train_num_correct/train_total_number
                    voss = val_num_correct/val_total_number
                elif(args.type == 'R'):
                    toss = toss / train_set.num_batches()
                    voss = voss / val_set.num_batches()
                print(f"Update %06d: train=%.3f val=%.3f" % (i, toss, voss))

        # resets values for accuracy calculation
        val_num_correct = 0
        val_total_number = 0

        if(args.v != True): # if not in verbose, runs through val once each epoch
            for i in range(val_set.num_batches()):
                x, y = val_set[i]

                if(args.type=='C'): 
                    y = one_hot(y, args.num_classes)
                Zs, As, yhat = forward(args, x, weights, biases)
                voss += compute_loss(args, y, yhat)

                if(args.type=='C'):     
                    num_correct, total, batch_acc, M = accuracy(y, yhat)
                    val_num_correct += num_correct
                    val_total_number += total

        if(args.type == 'C'):
            toss = train_num_correct/train_total_number
            voss = val_num_correct/val_total_number
        elif(args.type == 'R'):
            toss = toss / train_set.num_batches()
            voss = voss / val_set.num_batches()
        print(f"Epoch %03d: train=%.3f val=%.3f" % (epoch, toss, voss))
            
# Dataloader class
class Dataloader:
    def __init__(self, features, targets, batch_size):
        self.features = features
        self.targets = targets
        self.batch_size = batch_size
        self.feature_len = len(features[0])
        self.num_samples = len(features)
        self._shuffle_data()

    def __len__(self):
        return len(self.features)

    # shuffles data 
    def _shuffle_data(self):
#        random.seed(32) # set seed for reproducability
        indicies = list(range(len(self.features)))
        random.shuffle(indicies)
        self.features = np.array(self.features)[indicies]
        self.targets = np.array(self.targets)[indicies]
    
    # gets a batch of data
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = start_index + self.batch_size
        if(self.batch_size == 0): # full batch training 
            end_index = -1
        return self.features[start_index:end_index].T, self.targets[start_index:end_index].T

    # calculates the number of batches within the dataset
    def num_batches(self):
        if(self.batch_size == 0): # full batch training
            self.batch_size = self.num_samples
        return self.num_samples//self.batch_size

# parses and returns command line arguments
def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='store_true', default=False, help='verbose mode')
    parser.add_argument('-train_feat', type=str, help='training feature file')
    parser.add_argument('-train_target', type=str, help='training target file')
    parser.add_argument('-dev_feat', type=str, help='dev feature file')
    parser.add_argument('-dev_target', type=str, help='dev target file')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train [default: 100]')
    parser.add_argument('-learnrate', type=float, default=0.01, help='learning rate [default: 0.001]')
    parser.add_argument('-nunits', type=int, default=10, help='number of hidden units [default: 10]')
    parser.add_argument('-type', type=str, default='R', help='problem type in {R, C} [default: R]')
    parser.add_argument('-hidden_act', type=str, default='relu', help='activation function for hidden units in {tanh, relu, sigmoid} [default: relu]')
    parser.add_argument('-init_range', type=float, default=0.1, help='initialization range for weights [default: 0.1]')
    parser.add_argument('-num_classes', type=int, default=2, help='number of classes for classification [default: 2]') 
    parser.add_argument('-mb', type=int, default=1, help='minibatch size [default: 10]')
    parser.add_argument('-nlayers', type=int, default=1, help='number of hidden layers [default: 1]') 
    return parser.parse_args()
    
if __name__ == '__main__':
    args = parse_all_args()
    main(args)
