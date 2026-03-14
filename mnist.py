import cv2
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

# ============================================================
# DATA
# ============================================================

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# NN: beholder 28x28 form, normaliserer til 0-1
x_train_nn = x_train / 255.0
x_test_nn  = x_test  / 255.0

# KNN og RF: flater ut til 784 piksler, normaliserer til 0-1
x_train_sk = x_train.reshape(-1, 784) / 255.0
x_test_sk  = x_test.reshape(-1, 784)  / 255.0


# ============================================================
# TRENE MODELLER
# ============================================================

# Neural Network
print('Trener Neural Network...')
nn_model = tf.keras.models.Sequential()
nn_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
nn_model.add(tf.keras.layers.Dense(128, activation='relu'))
nn_model.add(tf.keras.layers.Dense(128, activation='relu'))
nn_model.add(tf.keras.layers.Dense(10, activation='softmax'))
nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(x_train_nn, y_train, epochs=3)
nn_model.save('neural_network/mnist/models/handwritten.keras')

# KNN
print('Trener KNN...')
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(x_train_sk, y_train)
joblib.dump(knn_model, 'neural_network/mnist/models/knn_model.pkl')

# Random Forest
print('Trener Random Forest...')
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train_sk, y_train)
joblib.dump(rf_model, 'neural_network/mnist/models/random_forest_model.pkl')


# ============================================================
# EVALUERE
# ============================================================

loss_nn, accuracy_nn = nn_model.evaluate(x_test_nn, y_test)
accuracy_knn = knn_model.score(x_test_sk, y_test)
accuracy_rf  = rf_model.score(x_test_sk, y_test)

y_pred_nn  = np.argmax(nn_model.predict(x_test_nn), axis=1)
y_pred_knn = knn_model.predict(x_test_sk)
y_pred_rf  = rf_model.predict(x_test_sk)

print(f'\nNeural Network  — Loss: {loss_nn:.4f} | Accuracy: {accuracy_nn:.4f}')
print(f'KNN             — Accuracy: {accuracy_knn:.4f}')
print(f'Random Forest   — Accuracy: {accuracy_rf:.4f}')


# ============================================================
# ACCURACY BAR CHART
# ============================================================

models     = ['Neural Network', 'KNN', 'Random Forest']
accuracies = [accuracy_nn, accuracy_knn, accuracy_rf]
colors     = ['steelblue', 'darkorange', 'forestgreen']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, accuracies, color=colors)
ax.set_ylim(0.9, 1.0)
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f'{acc:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()


# ============================================================
# CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_nn)).plot(cmap='Blues', ax=axes[0])
axes[0].set_title('Neural Network')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn)).plot(cmap='Oranges', ax=axes[1])
axes[1].set_title('KNN')
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(cmap='Greens', ax=axes[2])
axes[2].set_title('Random Forest')

plt.tight_layout()
plt.show()


# ============================================================
# TEGNE EGET TALL
# ============================================================

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
        img_nn = small.reshape(1, 28, 28)
        img_sk = small.reshape(1, -1)

        nn_probabilities  = nn_model.predict(img_nn)
        knn_probabilities = knn_model.predict_proba(img_sk)
        rf_probabilities  = rf_model.predict_proba(img_sk)

        nn_prediction  = np.argmax(nn_probabilities)
        knn_prediction = knn_model.predict(img_sk)[0]
        rf_prediction  = rf_model.predict(img_sk)[0]

        print(f'Neural Network Prediction: {nn_prediction}')
        print(f'KNN Prediction: {knn_prediction}')
        print(f'Random Forest Prediction: {rf_prediction}')

        plt.imshow(small, cmap='gray')
        plt.title(f'NN: {nn_prediction} | KNN: {knn_prediction} | RF: {rf_prediction}')
        plt.show()

        user_input = input('Analyze another image? (y/n): ')
        if user_input.lower() != 'y':
            break