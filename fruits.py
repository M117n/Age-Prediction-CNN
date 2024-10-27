import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AvgPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

optimizer = Adam()

def load_data(path):
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, validation_generator

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

def train_model(model, train_generator, validation_generator, epochs=5):
    model.fit(
        train_generator,
        epochs=epochs,
        verbose=2,
        validation_data=validation_generator
    )

    return model