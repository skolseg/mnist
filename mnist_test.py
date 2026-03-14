import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# MNIST datasett
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test_nn = tf.keras.utils.normalize(x_test, axis=1) # normaliser testdata for nn modellen
x_test_knn = x_test.reshape(-1, 28*28) / 255 # flater ut og normaliserer
x_test_rf = x_test.reshape(-1, 28*28) / 255


# laste modeller
nn_model = tf.keras.models.load_model('neural_network/mnist/models/handwritten.keras')
knn_model = joblib.load('neural_network/mnist/models/knn_model.pkl')
rf_model = joblib.load('neural_network/mnist/models/random_forest_model.pkl')

# Neural Network ------------
loss_nn, accuracy_nn = nn_model.evaluate(x_test_nn, y_test) # evaluere modellen på testdata
y_pred_nn = np.argmax(nn_model.predict(x_test_nn), axis=1) # prediksjoner for confusion matrix


print('Neural Network Model:')

print(f'Loss:\n', loss_nn)
print('--------------------------------')
print(f'Accuracy:\n', accuracy_nn)
print('--------------------------------')

print('')

# KNN ------------
accuracy_knn = knn_model.score(x_test_knn, y_test) # evaluer modellen på testdata
y_pred_knn = knn_model.predict(x_test_knn) # prediksjoner for confusion matrix

print('KNN Model:')

print(f'Accuracy:\n', accuracy_knn)
print('--------------------------------')

print('')

# Random Forest ------------
accuracy_rf = rf_model.score(x_test_rf, y_test) # evaluer modellen på testdata
y_pred_rf = rf_model.predict(x_test_rf) # prediksjoner for confusion matrix

print('Random Forest Model:')

print(f'Accuracy:\n', accuracy_rf)
print('--------------------------------')

print('')

# Confusion Matrix
print('Confusion Matrices:')

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nn)).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Neural Network')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn)).plot(cmap='Oranges', ax=axes[1])
axes[1].set_title('KNN')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(cmap='Greens', ax=axes[2])
axes[2].set_title('Random Forest')

plt.tight_layout()
plt.show()








print('')

# Analysere et bilde
while True:
    analyze_image = input('Analyze your own image? (y/n): ')
    if analyze_image.lower() in ('y', 'n'):
        break
    print('Invalid input. Please enter "y" or "n".')
while True:
    if analyze_image.lower() == 'y':

        SIZE = 280
        canvas = np.zeros((SIZE, SIZE))
        drawing = False

        fig, ax = plt.subplots()
        img_display = ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Draw your number and close window when done.')

        def on_press(event):
            global drawing
            drawing = True

        def on_release(event):
            global drawing
            drawing = False

        RADIUS = 20

        def on_move(event):
            if drawing and event.xdata and event.ydata:
                x, y = int(event.xdata), int(event.ydata)
                for dx in range(-RADIUS, RADIUS + 1):
                    for dy in range(-RADIUS, RADIUS + 1):
                        dist = (dx**2 + dy**2) ** 0.5
                        if dist <= RADIUS:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                                strength = 1.0 - (dist / RADIUS)
                                canvas[ny, nx] = min(1.0, canvas[ny, nx] + strength)
                img_display.set_data(canvas)
                fig.canvas.draw()

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_move)

        plt.show()

        small = cv2.resize(canvas, (28, 28))
        img_nn = tf.keras.utils.normalize(small.reshape(1, 28, 28), axis=1)
        img_knn = (small).reshape(1, -1)
        img_rf = (small).reshape(1, -1)

        nn_probabilities = nn_model.predict(img_nn)
        knn_probabilities = knn_model.predict_proba(img_knn)
        rf_probabilities = rf_model.predict_proba(img_rf)

        nn_prediction = np.argmax(nn_probabilities)
        knn_prediction = knn_model.predict(img_knn)[0]
        rf_prediction = rf_model.predict(img_rf)[0]

        print(f'Neural Network Prediction: {nn_prediction}')
        print(f'KNN Prediction: {knn_prediction}')
        print(f'Random Forest Prediction: {rf_prediction}')
        print('--------------------------------')
        print(f'Neural Network Probabilities: {nn_probabilities}')
        print(f'KNN Probabilities: {knn_probabilities}')
        print(f'Random Forest Probabilities: {rf_probabilities}')


        plt.imshow(small, cmap='gray')
        plt.title(f'NN: {nn_prediction} | KNN: {knn_prediction} | RF: {rf_prediction}')
        plt.show()

        user_input = input('Analyze another image? (y/n): ')
        if user_input.lower() != 'y':
            break
