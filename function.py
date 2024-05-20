import numpy as np

def relu(inputs):
    return np.maximum(inputs, 0)

def softmax(inputs):
    exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(inputs, labels):
    indices = np.argmax(labels, axis=1)
    probs = inputs[np.arange(len(inputs)), indices]
    log_probs = -np.log(probs)
    return np.mean(log_probs)

def L2_regularization(lambda_, *weights):
    return 0.5 * lambda_ * sum(np.sum(w ** 2) for w in weights)
