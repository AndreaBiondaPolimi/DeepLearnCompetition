import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam


def lenet_mod_model(input_shape, output_shape):

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=80, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=112, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=144, kernel_size=(3,3), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))

    model.summary()
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

    return model

def train (model, train_dataset, valid_dataset, epochs):
    model.fit(x=train_dataset,
          epochs=100,  #### set repeat in training dataset
          steps_per_epoch=30,
          validation_data=valid_dataset,
          validation_steps=1)

    model.save('classification.h5')
