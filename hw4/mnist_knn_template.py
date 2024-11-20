
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
plt.show()


# Data preprocessing
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)


# KNN classifier
def distance(x,y):
    # define some distance function
    return d

def kNN(x, k, data, label):
    #list of distances between the given image and the images of the training set
    distances = [distance(x,data[i]) for i in range(len(data))]
    return clas # estimated class

def image_show(i, data, label, clas):
    x = data[i] # get vectorized image
    x = x.reshape((28,28)) # reshape it into 28x28 format
    title = 'predicted={0:d}, true={0:d}'.format(clas, label[i])
    plt.imshow(x, cmap='gray') 
    plt.title(title)
    plt.show()


# Test case
i = 10
clas = kNN(X_test[i], k, X_train, y_train)
image_show(i, X_test, y_test, clas)

# Precision on test data
print('precision = ', precision)
    
