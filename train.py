# d7043

# Train the specified model with input Midi

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from time import time
import keras
import matplotlib.pyplot as plt


class Gru:
    def __init__(self, name, model_dir="models/"):
        self.model = None
        self.name = name
        self.model_dir = model_dir

    def build(self, lookback, num_pitches):
        # Build the model architecture
        model = Sequential()
        model.add(layers.LSTM(256, input_shape=(lookback, num_pitches), return_sequences=True, dropout=0.1, recurrent_dropout=0.2))
        model.add(layers.LSTM(128, dropout=0.1, recurrent_dropout=0.2, return_sequences=False))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_pitches, activation='sigmoid'))
        model.summary()
        print("Model built.")

        model.compile(loss='categorical_crossentropy',  # categorical_crossentropy or mse
                      optimizer=RMSprop(),
                      metrics=['acc'])
        self.model = model

    def set_model(self, model):
        print("Overwriting model -- Ensure that correct dimesions are being used")
        self.model = model

    def load(self, fname):
        self.model = load_model(fname)

    def save(self, fname):
        save_model(self.model, fname)

    def train(self, x, y, epochs, batch_size=128):
        begin = time()
        historys = []
        for i in range(epochs):
            print('>Epoch', i+1)
            tic = time()
            history = self.model.fit(x, y,
                                epochs=1,
                                batch_size=batch_size,
                                verbose=1)
            print('Took %s minutes for epoch %s. %s minutes elapsed.' % (
            ((time() - tic) / 60).__round__(2), i+1, ((time() - begin) / 60).__round__(2)))
            self.save(self.model_dir + self.name + ".h5")
            historys.append(history)
        print('Full train took %s minutes.' % ((time() - begin) / 60).__round__(2))
        return historys


def save_model(model, fname):
    model.save(fname)
    print('Saved model "' + fname + '"')


def load_model(fname):
    model = keras.models.load_model(fname)
    print('Loaded model "' + fname + '"')
    return model


def plot_history(history):
    acc = history.history['acc']
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
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
