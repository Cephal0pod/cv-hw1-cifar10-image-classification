import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, cache):
    x = cache
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(dout, cache):
    s = cache
    return dout * s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_backward(dout, cache):
    return dout * (1 - np.tanh(cache)**2)

activation_forward = {
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh
}

activation_backward = {
    'relu': relu_backward,
    'sigmoid': sigmoid_backward,
    'tanh': tanh_backward
}
