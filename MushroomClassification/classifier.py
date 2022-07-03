# dataset: https://www.kaggle.com/datasets/derekkunowilliams/mushrooms

import os
import shutil
from math import ceil

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from keras.callbacks import CSVLogger

import pandas as pd
import matplotlib.pyplot as plt


def move_files(path1, path2):
    for i, x in enumerate(os.walk(path1)):
        if i > 0:
            last = x[0].split('\\')[-1]
            filenames = next(os.walk(x[0]), (None, None, []))[2]
            size = len(filenames)
            first_ind = ceil(size * 0.70)
            testing = filenames[first_ind:]
            for name in testing:
                shutil.move(x[0] + '\\' + name, path2 + last + '\\' + name)
            # files_poisonous[last] = filenames[first_ind:]
            # os.mkdir('dataset\\testing\\poisonous\\' + last)
            # print(last)


def count_files():
    noOfFiles = 0
    for base, dirs, files in os.walk('dataset\\training\\poisonous'):
        print('Looking in : ', base)
        for Files in files:
            noOfFiles += 1
    print('Number of files', noOfFiles)


def load_train_data():
    TRAIN_PATH = 'dataset\\training\\'
    train = ImageDataGenerator(rescale=1 / 255)
    train_dataset = train.flow_from_directory(TRAIN_PATH, batch_size=100, class_mode='categorical', target_size=(100, 100))
    return train_dataset


def load_test_data():
    TEST_PATH = 'dataset\\testing\\'
    test = ImageDataGenerator(rescale=1 / 255)
    test_dataset = test.flow_from_directory(TEST_PATH, batch_size=100, class_mode='categorical', target_size=(100, 100))
    return test_dataset


def train_model():
    train_data = load_train_data()
    test_data = load_test_data()
    model = Sequential([

        # convolutional
        Conv2D(32, kernel_size=2, activation='relu', input_shape=(100, 100, 3), padding='same'),
        MaxPool2D(2, 2),
        Dropout(0.1),
        Conv2D(64, kernel_size=2, activation='relu', padding='same'),
        MaxPool2D(2, 2),
        Dropout(0.1),

        # dense
        Flatten(),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    csv_logger = CSVLogger('training.log', separator=',', append=False)
    model.fit(train_data, epochs=100, validation_data=test_data, callbacks=[csv_logger])


if __name__ == '__main__':
    # train_model()
    log_data = pd.read_csv('training.log', sep=',', engine='python')
    epochs = log_data['epoch']
    accuracies = log_data['accuracy']
    losses = log_data['loss']
    print(log_data['accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(epochs, accuracies)
    plt.show()
