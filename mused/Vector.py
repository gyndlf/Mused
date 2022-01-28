""" d7154

Class for Vector

Trys to predict if the music is authentic or made by Gru

Used once Gru can make decent music, and to simulate some forwards thinking

"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from .functions import save_model, load_model


class Vector:
    """The Vector model to classify the type of music"""
    def __init__(self, name, model_dir='out/vector/'):
        self.name = name
        self.model_dir = model_dir
        self.model = None

    def set_model(self, model):
        self.model = model

    def load(self, fname):
        self.model = load_model(fname)

    def save(self, fname):
        save_model(self.model, fname)

    def build(self, lookback, num_pitches, loss='binary_crossentropy',):
        """Build a new model"""
        input_layer = layers.Input(shape=(lookback, num_pitches))
        conv1 = layers.Conv1D(filters=32, kernel_size=8, strides=1, activation='relu', padding='same')(input_layer)
        lstm1 = layers.LSTM(256, return_sequences=True)(conv1)
        lstm2 = layers.LSTM(512, return_sequences=True)(lstm1)
        lstm3 = layers.LSTM(256, return_sequences=False)(lstm2)
        dropout1 = layers.Dropout(0.3)(lstm3)
        dense1 = layers.Dense(256, activation='relu')(dropout1)
        dropout2 = layers.Dropout(0.3)(dense1)
        output_layer = layers.Dense(1, activation='sigmoid')(dropout2)
        model = models.Model(inputs=input_layer, outputs=output_layer)

        model.summary()
        print("Vector model built.")

        model.compile(loss=loss,  # binary_crossentropy
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=["accuracy", "mean_absolute_error"])
        self.model = model

    def train(self):
        ...