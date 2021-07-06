# d7043

# Train the specified model with input Midi
# Training model functions

from time import time
import numpy as np
import tensorflow as tf
from tensorflow import keras  # import from tensorflow for better support??? I dunno
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
import generate


class GenerateMusic(tf.keras.callbacks.Callback):
    """Callback to generate music during training"""
    def __init__(self, gen_roll, gen_every=5, length=24*4*2, threshold=0.7):
        super(GenerateMusic, self).__init__()
        self.generated = {}
        self.gen_every = gen_every
        self.roll = gen_roll
        self.length = length
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.gen_every == 0:
            # Generate some music
            self.generated["epoch-" + str(epoch + 1)] = generate.generate_music(self.model, self.roll, 0.8, length=self.length, threshold=self.threshold)

    def on_train_end(self, logs=None):
        self.generated["last-generation"] = generate.generate_music(self.model, self.roll, 0.8, length=self.length, threshold=self.threshold)
        generate.multi_save(self.generated, "outputs/generated-during-training.mid")


class Gru:
    """The controlling class for the training model"""
    def __init__(self, name, model_dir="models/"):
        self.model = None
        self.name = name
        self.model_dir = model_dir

    def build(self, lookback, num_pitches, loss='mse'):
        # Build the model architecture
        model = models.Sequential()
        model.add(layers.LSTM(128, input_shape=(lookback, num_pitches), return_sequences=False,
                              dropout=0.1, recurrent_dropout=0.2))
        # model.add(layers.LSTM(128, dropout=0.1, recurrent_dropout=0.2, return_sequences=False))
        model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_pitches, activation='sigmoid'))
        model.summary()
        print("Model built.")

        model.compile(loss=loss,  # categorical_crossentropy or mse
                      optimizer=keras.optimizers.RMSprop(learning_rate=1e-04),
                      metrics=["acc", "mean_absolute_error"])
        self.model = model

    def set_model(self, model):
        print("Overwriting model -- Ensure that correct dimesions are being used")
        self.model = model

    def load(self, fname):
        self.model = load_model(fname)

    def save(self, fname):
        save_model(self.model, fname)

    def train(self, x, y, epochs, callbacks, batch_size=128):
        historys = []  # Kept for legacy purposes

        tic = time()
        history = self.model.fit(x, y,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=1,
                                 callbacks=callbacks)

        self.save(self.model_dir + self.name + ".h5")  # Save the model
        print('Full train took %s minutes.' % ((time() - tic) / 60).__round__(2))
        return historys


def save_model(model, fname):
    model.save(fname)
    print('Saved model "' + fname + '"')


def load_model(fname):
    model = models.load_model(fname)
    print('Loaded model "' + fname + '"')
    return model


def plot_history(historys):
    acc = []
    loss = []

    for history in historys:
        acc.append(history.history['acc'])
        loss.append(history.history['loss'])

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'b-', label='Validation acc')
    plt.title('Training accuracy')
    plt.legend()

    plt.figure()  # Combines the two graphs

    plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()
