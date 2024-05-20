import numpy as np
from network import Network
import mnist

# Load data
num_classes = 10
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print("Training...")

# Data processing
X_train = train_images.reshape(train_images.shape[0], -1).astype('float32') / 255
y_train = np.eye(num_classes)[train_labels]

X_test = test_images.reshape(test_images.shape[0], -1).astype('float32') / 255
y_test = test_labels

net = Network(
    num_nodes_in_layers=[784, 64, 128, 10],  # Updated for 4 layers: input, 2 hidden, output
    batch_size=1,
    num_epochs=7,
    learning_rate=0.001,
    weights_file='filename.pkl'
)

net.train(X_train, y_train)

print("Testing...")
net.test(X_test, y_test)
