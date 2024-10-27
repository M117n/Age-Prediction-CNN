from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
import numpy as np

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
   base_model = ResNet50(input_shape=(150, 150, 3), weights='imagenet', include_top=False)
   model = Sequential()
   model.add(base_model)
   model.add(GlobalAveragePooling2D())
   model.add(Dense(units=12, activation='softmax'))

   return model

def train_model(model, train_generator, validation_generator, epochs=3):
    model.fit(
        train_generator,
        epochs=epochs,
        verbose=2,
        validation_data=validation_generator
    )

    return model