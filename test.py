import numpy as np
import pickle
import cv2
import mnist
import function

# Load data
test_images = mnist.test_images()
test_labels = mnist.test_labels()

X_test = test_images.reshape(test_images.shape[0], -1).astype('float32') / 255

with open('filename.pkl', 'rb') as f:
    weight1, bias1, weight2, bias2, weight3, bias3 = pickle.load(f)

for num in range(len(test_images)):
    z1 = np.dot(X_test[num:num+1], weight1) + bias1
    a1 = function.relu(z1)
    z2 = np.dot(a1, weight2) + bias2
    a2 = function.relu(z2)
    z3 = np.dot(a2, weight3) + bias3
    y = function.softmax(z3)
    predict = np.argmax(y)

    img = np.stack([test_images[num]]*3, axis=-1)
    resized_image = cv2.resize(img, (500, 500))
    cv2.putText(resized_image, str(predict), (5, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)
    cv2.imshow('input', resized_image)
    
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
