import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist

# laste inn data fra MNIST-datasettet
# x er piksler, y er klassifisering av tallene
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normaliser dataen slik at den er mellom 0 og 1
x_train = tf.keras.utils.normalize(x_train, axis=1) # axis=1 normaliserer hver rad
x_test = tf.keras.utils.normalize(x_test, axis=1)

# bygge modellen
model = tf.keras.models.Sequential() # sekvensiell modell, legger til lag etter lag
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # flater ut 28x28 bildene til en 782 piksler lang ting
model.add(tf.keras.layers.Dense(128, activation='relu')) # fullt tilkoblet lag med 128 noder, relu aktiveringsfunksjon
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) # 10 noder for 10 klasser, softmax for sannsynligheter

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # kompilere modellen med adam optimizer og tapfunksjon

model.fit(x_train, y_train, epochs=3) # trene modellen i 3 epoker

model.save('neural_network/mnist/models/handwritten.keras') # lagre modellen

loss, accuracy = model.evaluate(x_test, y_test) # evaluere modellen på testdata
print(f'Loss:\n', loss)
print('--------------------------------')
print(f'Accuracy:\n', accuracy)