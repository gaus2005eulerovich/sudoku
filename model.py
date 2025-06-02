# Установка зависимостей
!pip install opencv-python-headless imutils scikit-image tensorflow
!sudo apt install tesseract-ocr
!pip install pytesseract

# Импорт библиотек
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
import imutils
import cv2
import os
from google.colab import files
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# Функция для создания модели
def build_model(input_shape, classes):
    model = Sequential()
    # Первый набор слоёв: CONV => RELU => POOL
    model.add(Conv2D(32, (5, 5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Второй набор слоёв: CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Первый набор FC => RELU слоёв
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # Второй набор FC => RELU слоёв
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # Классификатор softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model

# Загрузка и подготовка данных MNIST
print("[INFO] Загрузка MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# Подготовка данных
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# One-hot кодирование меток
trainY_cat = to_categorical(trainY, 10)
testY_cat = to_categorical(testY, 10)

# Создание и компиляция модели
inputShape = (28, 28, 1)
classes = 10
model = build_model(inputShape, classes)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Обучение модели
print("[INFO] Обучение модели...")
H = model.fit(trainX, trainY_cat,
              validation_data=(testX, testY_cat),
              batch_size=128,
              epochs=10,
              verbose=1)

# Функция для аугментации проблемных цифр
def augment_digits(images, labels):
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        # Особенно важны 6 и 8
        if label in [6, 8]:
            for _ in range(5):  # Создаем 5 вариаций для каждой цифры
                # Случайное вращение
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                transformed = cv2.warpAffine(img.squeeze(), M, (28, 28), flags=cv2.INTER_CUBIC)
                transformed = transformed.reshape(28, 28, 1)
                augmented_images.append(transformed)
                augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

# Дообучение модели на проблемных цифрах
print("[INFO] Дообучение модели на проблемных цифрах...")
trainX_aug, trainY_aug = augment_digits(trainX, trainY)
trainY_aug_cat = to_categorical(trainY_aug, 10)

model.fit(trainX_aug, trainY_aug_cat,
          batch_size=32,
          epochs=3,
          verbose=1)

# Оценка модели
print("[INFO] Оценка модели...")
(loss, accuracy) = model.evaluate(testX, testY_cat, verbose=0)
print(f"Точность на тестовых данных: {accuracy*100:.2f}%")

# Сохранение модели
model.save("sudoku_model_v2.h5")
print("[INFO] Модель сохранена как 'sudoku_model_v2.h5'")
