import numpy as np
import tensorflow as tf
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

mnist = tf.keras.datasets.mnist

# laste inn data fra MNIST-datasettet
# x er piksler, y er klassifisering av tallene
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255 # flater ut bildene til en 784 piksler lang rad
x_test = x_test.reshape(-1, 28*28) / 255

rf = RandomForestClassifier(n_estimators=100) # opprett Random Forest-modellen med 100 trær

rf.fit(x_train, y_train) # tren modellen på treningsdata

joblib.dump(rf, 'neural_network/mnist/models/random_forest_model.pkl') # lagre modellen

accuracy_rf = rf.score(x_test, y_test) # evaluer modellen på testdata
print(f'Random Forest Accuracy:\n', accuracy_rf)