import numpy as np
from activation_functions import activation_forward, activation_backward
import matplotlib.pyplot as plt
import matplotlib

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 activation='relu', weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg
        self.activation = activation
        self.loss_history = []  # 存储每次训练过程中的 loss
        self.val_accuracy_history = []  # 存储每个 epoch 的验证准确率

        self.params['W1'] = weight_scale * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_scale * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        a1 = X.dot(W1) + b1
        if self.activation == 'relu':
            h1 = np.maximum(0, a1)
        else:
            h1 = activation_forward[self.activation](a1)
        scores = h1.dot(W2) + b2

        if y is None:
            return scores

        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = X.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        grads = {}
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads['W2'] = h1.T.dot(dscores) + self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        dh1 = dscores.dot(W2.T)
        if self.activation == 'relu':
            da1 = dh1 * (a1 > 0)
        else:
            da1 = activation_backward[self.activation](dh1, a1)
        grads['W1'] = X.T.dot(da1) + self.reg * W1
        grads['b1'] = np.sum(da1, axis=0)

        self.loss_history.append(loss)
        return loss, grads

    def predict(self, X):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        a1 = X.dot(W1) + b1
        h1 = np.maximum(0, a1)
        scores = h1.dot(W2) + b2
        return np.argmax(scores, axis=1)

    def plot_loss_and_accuracy(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Iterations/Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracy_history, label='Validation Accuracy', color='g')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
