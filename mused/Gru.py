# d7154

# Class for Gru.
# (Moved from train.py)

from tensorflow import keras  # import from tensorflow for better support??? I dunno
from tensorflow.keras import layers
from tensorflow.keras import models
from time import time
from .functions import save_model, load_model


class Gru:
    """The controlling class for the training model"""
    def __init__(self, name, model_dir="out/models/"):
        self.model = None
        self.name = name
        self.model_dir = model_dir

    def build(self, lookback, num_pitches, loss='binary_crossentropy'):
        # Build the model architecture
        model = models.Sequential()
        model.add(layers.LSTM(256, input_shape=(lookback, num_pitches), return_sequences=True,
                             dropout=0.3, recurrent_dropout=0.3))
        model.add(layers.LSTM(512, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
        model.add(layers.LSTM(256, return_sequences=False))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_pitches, activation='sigmoid'))

        # input_layer = layers.Input(shape=(lookback, num_pitches, 1))
        # conv1 = layers.Conv2D(32, (3,3), strides=1, activation='relu', padding='same')(input_layer)
        # max1 = layers.MaxPooling2D((1,1))(conv1)
        # reshape1 = layers.Reshape((lookback, num_pitches))(max1)
        # lstm1 = layers.LSTM(256, return_sequences=True)(reshape1)
        # lstm2 = layers.LSTM(512, return_sequences=True)(lstm1)
        # lstm3 = layers.LSTM(256, return_sequences=False)(lstm2)
        # dropout1 = layers.Dropout(0.3)(lstm3)
        # dense1 = layers.Dense(256, activation='relu')(dropout1)
        # dropout2 = layers.Dropout(0.3)(dense1)
        # output_layer = layers.Dense(num_pitches, activation='sigmoid')(dropout2)
        # model = models.Model(inputs=input_layer, outputs=output_layer)

        model.summary()
        print("Model built.")

        model.compile(loss=loss,  # categorical_crossentropy or mse or binary_crossentropy
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=["accuracy", "mean_absolute_error"])
        self.model = model

    def set_model(self, model):
        print("Overwriting model -- Ensure that correct dimensions are being used")
        self.model = model

    def load(self, fname):
        self.model = load_model(fname)

    def save(self, fname):
        save_model(self.model, fname)

    def train(self, x, y, epochs, callbacks, batch_size=128):
        tic = time()
        history = self.model.fit(x, y,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=1,
                                 callbacks=callbacks)

        historys = [history]  # Kept for legacy purposes
        self.save(self.model_dir + self.name + ".h5")  # Save the model
        print('Full train took %s minutes.' % ((time() - tic) / 60).__round__(2))
        return historys
