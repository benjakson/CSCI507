import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]])) 
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(5):
    img = X_train[i]
    axarr[i].imshow(img)
plt.show()


# Data preprocessing
X_train_orig = np.copy(X_train)
X_test_orig = np.copy(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], -1)) 
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# do more pre-processing as needed

# SVM classifier
from sklearn import svm
clf = svm.SVC()

# Evaluate on test set  
predicted = clf.predict(X_test)
score = clf.score(X_test,y_test) #classification score

# Test case
i = 10       
xVal = X_test[i, :]
yVal = y_test[i]   
yHat = predicted[i]
xImg = X_test_orig[i]
plt.imshow(xImg)
title = 'true={0:s} est={1:s}'.format(cifar_classes[yVal[0]], cifar_classes[yHat.astype(int)])
plt.title(title)
plt.show()
