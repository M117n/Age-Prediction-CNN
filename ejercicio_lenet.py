from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
import numpy as np

optimizer = Adam()

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(features_train.shape[0], 28, 28, 1) / 255.0
    
    return features_train, target_train

def create_model(input_shape):
    model = Sequential()

    model.add(
        Conv2D(
            filters=6,
            kernel_size=(5, 5),
            padding='same',
            activation="tanh",
            input_shape=input_shape,
        )
    )
    model.add(
        AvgPool2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        Conv2D(
            filters=16,
            kernel_size=(5, 5),
            padding='valid',
            activation="tanh",
        )
    )
    model.add(
        AvgPool2D(
            pool_size=(2, 2)
        )
    )
    model.add(
        Flatten()
    )
    model.add(
        Dense(
            units=120,
            activation='tanh'
        )
    )
    model.add(
        Dense(
            units=84,
            activation='tanh'
        )
    )
    model.add(
        Dense(
            units=10,
            activation='softmax'
        )
    )
    model.compile(
        loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc']
    )

    return model

def train_model(model, train_data, test_data):
    features_train, target_train = train_data
    features_test, target_test = test_data

    model.fit(
        features_train,
        target_train,
        epochs=5,
        verbose=2,
        batch_size=32,
        shuffle = True,
    )

    return model