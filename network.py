import numpy as np
import pickle
import function

class Network:
    def __init__(self, num_nodes_in_layers, batch_size, num_epochs, learning_rate, weights_file):
        self.num_nodes_in_layers = num_nodes_in_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights_file = weights_file
        
        # Initialize weights and biases for 4 layers
        self.weight1 = np.random.normal(0, 1, [self.num_nodes_in_layers[0], self.num_nodes_in_layers[1]])
        self.bias1 = np.zeros((1, self.num_nodes_in_layers[1]))
        self.weight2 = np.random.normal(0, 1, [self.num_nodes_in_layers[1], self.num_nodes_in_layers[2]])
        self.bias2 = np.zeros((1, self.num_nodes_in_layers[2]))
        self.weight3 = np.random.normal(0, 1, [self.num_nodes_in_layers[2], self.num_nodes_in_layers[3]])
        self.bias3 = np.zeros((1, self.num_nodes_in_layers[3]))
        self.loss = []

    def train(self, inputs, labels):
        for epoch in range(self.num_epochs):
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
            inputs = inputs[indices]
            labels = labels[indices]

            for i in range(0, len(inputs), self.batch_size):
                inputs_batch = inputs[i:i+self.batch_size]
                labels_batch = labels[i:i+self.batch_size]

                # Forward pass
                z1 = np.dot(inputs_batch, self.weight1) + self.bias1
                a1 = function.relu(z1)
                z2 = np.dot(a1, self.weight2) + self.bias2
                a2 = function.relu(z2)
                z3 = np.dot(a2, self.weight3) + self.bias3
                y = function.softmax(z3)

                # Calculate loss
                loss = function.cross_entropy(y, labels_batch)
                loss += function.L2_regularization(0.01, self.weight1, self.weight2, self.weight3)
                self.loss.append(loss)

                # Backward pass
                delta_y = (y - labels_batch) / y.shape[0]
                delta_a2 = np.dot(delta_y, self.weight3.T)
                delta_a2[a2 <= 0] = 0
                delta_a1 = np.dot(delta_a2, self.weight2.T)
                delta_a1[a1 <= 0] = 0

                # Gradient calculation
                weight3_gradient = np.dot(a2.T, delta_y)
                bias3_gradient = np.sum(delta_y, axis=0, keepdims=True)
                weight2_gradient = np.dot(a1.T, delta_a2)
                bias2_gradient = np.sum(delta_a2, axis=0, keepdims=True)
                weight1_gradient = np.dot(inputs_batch.T, delta_a1)
                bias1_gradient = np.sum(delta_a1, axis=0, keepdims=True)

                # L2 regularization
                weight3_gradient += 0.01 * self.weight3
                weight2_gradient += 0.01 * self.weight2
                weight1_gradient += 0.01 * self.weight1

                # Stochastic gradient descent update
                self.weight1 -= self.learning_rate * weight1_gradient
                self.bias1 -= self.learning_rate * bias1_gradient
                self.weight2 -= self.learning_rate * weight2_gradient
                self.bias2 -= self.learning_rate * bias2_gradient
                self.weight3 -= self.learning_rate * weight3_gradient
                self.bias3 -= self.learning_rate * bias3_gradient

                if i % 100 == 0:
                    print(f'Epoch {epoch+1}/{self.num_epochs}, Iteration {i}, Loss: {loss:.4f}')

        with open(self.weights_file, 'wb') as f:
            pickle.dump([self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3], f)

    def test(self, inputs, labels):
        z1 = np.dot(inputs, self.weight1) + self.bias1
        a1 = function.relu(z1)
        z2 = np.dot(a1, self.weight2) + self.bias2
        a2 = function.relu(z2)
        z3 = np.dot(a2, self.weight3) + self.bias3
        y = function.softmax(z3)
        accuracy = np.mean(np.argmax(y, axis=1) == labels)
        print(f'Test accuracy: {accuracy * 100:.2f}%')
