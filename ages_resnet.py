import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

optimizer =  Adam()

#labels = pd.read_csv('/datasets/faces/labels.csv')
#labels['file_path'] = '/datasets/faces/final_files/' + labels['file_name']

#Hacemos aumento en los datos para introducir variabilidad y mejorara la capacidad de generalizacion del modelo
datagen = ImageDataGenerator(validation_split=0.25,
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

def load_data(path, subset):
    train_gen = datagen.flow_from_directory(directory=path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='raw',
        subset='training')

    test_gen = datagen.flow_from_directory(directory=path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='raw',
        subset='validation')

    return train_gen, test_gen

def create_model(input_shape):
    base_model = ResNet50(input_shape=input_shape,
                          weights='imagenet',
                          include_top=False)
    
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_absolute_error', 
                  optimizer=optimizer,
                  metrics=['mae'])

    return model

def train_model(model, train_data, test_data, batch_size=32, steps_per_epoch=None, validation_steps=None, epochs=3,):
    model.fit(train_data,
        epochs=epochs,
        verbose=2,
        validation_data=test_data,
        validation_steps=validation_steps,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,)
    
    return model