import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.neighbors import KNeighborsClassifier
import joblib

mnist = tf.keras.datasets.mnist

# laste inn data fra MNIST-datasettet
# x er piksler, y er klassifisering av tallene
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28) / 255 # flater ut bildene til en 784 piksler lang rad, og normaliserer ved å dele på 255
x_test = x_test.reshape(-1, 28*28) / 255

knn = KNeighborsClassifier(n_neighbors=3) # opprett KNN-modellen med x naboer
knn.fit(x_train, y_train) # tren modellen på treningsdata

joblib.dump(knn, 'neural_network/mnist/models/knn_model.pkl') # lagre modellen

accuracy = knn.score(x_test, y_test) # evaluer modellen på testdata
print(f'KNN Accuracy:\n', accuracy)