# d6595
# Heavy rewrite on d7042

# The code containing the actual functions for the midi. Can be used across multiple model architectures
# Basic midi <-> numpy interface
import keras
import matplotlib.pyplot as plt
from tensorflow.keras import models

# TODO:
#  - Fix tempos. Make sure they are always the same
#  - Trim silence
#  - Force piece to be in the same key
#  - Convert from print statements to logging


def save_model(model, fname):
    """Save the input model"""
    model.save(fname)
    print('Saved model "' + fname + '"')


def load_model(fname) -> keras.Model:
    """Return the input model"""
    model = models.load_model(fname)
    print('Loaded model "' + fname + '"')
    return model


def plot_history(historys):
    """Plot the training histories"""
    acc = []
    loss = []

    for history in historys:
        acc.append(history.history['accuracy'])
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
