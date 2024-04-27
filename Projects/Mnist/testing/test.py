### NetWork ###
# Linear
# Reulu
# Linear
# Softmax

### Forwardprop ###
# A0 = X [784 x m]
# Z1 [10 x m] = W1 [10 x 748] . A0 [784 x m] + B [10 x m]
# A1 = ReLu(Z1) ### x if x > 0 | 0 if x<=0
# Z2 [10 x m] = W2 [10 x 748] . A1 [10 x m] + B [10 x m]
# A2 = softMax(Z2) (prob)

### BackProp ###
# dZ2 [10 x m] = A2 [10 x m] - Labels [10 x m]
# dW2 = 1/m . dZ2 . A1T [m x 10]
# dB2 = 1/m . sum(dZ2) [10 x 1]
# dZ1 = W2T [10 x 10] . dZ2 . g' [10 x m]
# dW1 = 1/m . dZ1 [10 x m] . XT [m x 784]
# dB1 = 1/m . sum(dZ1) [10 x 1]

### Learning ###
# W1 = W1 - a . dW1
# B1 = B1 - a . dB1
# W2 = W2 - a . dW2
# B2 = B2 - a . dB2

import numpy as np
from matplotlib import pyplot as plt

def load_mnist_images(file_path):
    """
    Load MNIST images from the given file path.
    """
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def load_mnist_labels(file_path):
    """
    Load MNIST labels from the given file path.
    """
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

### Load data ###
train_images = load_mnist_images("../data/train-images.idx3-ubyte")
train_label = load_mnist_labels("../data/train-labels.idx1-ubyte")

print(train_images.shape)
print(train_label.shape)

print(train_images[0])

### Split data ###
m , n = train_images.shape

X_dev = train_images[0:1000].T
Y_dev = train_label[0:1000]

X_train = train_images[1000:].T
Y_train = train_label[1000:]

m_train, n_train = X_train.shape

print(X_train)

## NN

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2

def ReLU(Z):
    return np.maximum(Z, 0)

def SoftMax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def deriv_ReLU(Z):
    return Z > 0

def OneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forwardProp(X, W1, B1, W2, B2):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = SoftMax(Z2)

    return Z1, A1, Z2, A2

def backProp(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = OneHot(Y)
    dZ2 = A2 - one_hot_Y

    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)

    dW1 = 1 / m * dZ1.dot(X.T)
    dB1 = 1 / m * np.sum(dZ1)

    return dW1, dB1, dW2, dB2

def updateParams(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2

    return W1, B1, W2, B2

def getPrediction(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradientDecent(X, Y , iterations, learnRate):
    W1, B1, W2, B2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(X, W1, B1, W2, B2)
        dW1, dB1, dW2, dB2 = backProp(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = updateParams(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learnRate)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", getAccuracy(getPrediction(A2), Y))

    return W1, B1, W2, B2

W1, B1, W2, B2 = gradientDecent(X_train, Y_train, 500, 0.1)