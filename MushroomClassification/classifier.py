import os
import shutil
from math import ceil

from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten


# cnn.fit: (x_train, y_train, epochs)
# training: 70%
# testing 30%
# 100, 100, 3

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


def load_train_data():
    train = ImageDataGenerator(rescale=1 / 255)
    train_dataset = train.flow_from_directory('dataset\\training\\', batch_size=100, class_mode='categorical', target_size=(100, 100))
    return train_dataset


def load_test_data():
    test = ImageDataGenerator(rescale=1 / 255)
    test_dataset = test.flow_from_directory('dataset\\testing\\', batch_size=100, class_mode='categorical', target_size=(100, 100))
    return test_dataset


if __name__ == '__main__':
    train_data = load_train_data()
    test_data = load_test_data()
    model = Sequential([
        # cnn
        # Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        # MaxPool2D(2, 2),
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
        MaxPool2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPool2D(2, 2),
        # Conv2D(128, (3, 3), activation='relu'),
        # MaxPool2D(2, 2),

        # dense
        Flatten(),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_fit = model.fit(train_data, epochs=15, validation_data=test_data)
