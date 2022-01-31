# d6595
# Heavy rewrite on d7042

# The code containing the actual functions for the midi. Can be used across multiple model architectures
# Basic midi <-> numpy interface
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
import pypianoroll
from .Midi import Midi

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


def extract_music(preds, temperature=1.0, threshold=None, noise=False):
    """Extract the new line from the predictions (Booleanise)"""
    preds = np.asarray(preds).astype('float64')  # Convert to np array
    preds = np.log(preds) / temperature
    preds = np.exp(preds)

    output = np.zeros(preds.shape)
    if noise:
        r = np.random.random(preds.shape)
        output[r < preds] = 1  # has to be below the critera
    elif threshold is None:
        raise Exception("Threshold should not be none with noise being false! One for the other!")
    else:
        output[threshold < preds] = 1  # Greater than 50% chance
    # p/p.sum(axis=0) (Normalise)
    return output.astype('bool')


def multi_save(piano_rolls, fname):
    """Saves multiple tracks as one midi, for easy use within logic"""
    midi = Midi(70)  # number 70 is a placeholder to be rewritten
    tracks = []
    for i, name in enumerate(piano_rolls):
        midi.load_np(piano_rolls[name])
        roll = midi.reformat_roll()
        t = pypianoroll.BinaryTrack(pianoroll=roll, program=0, is_drum=False, name='Generated-' + str(name))
        tracks.append(t)
    print('Saving', len(tracks), 'tracks in one file "' + fname + '".')
    mt = pypianoroll.Multitrack(tracks=tracks)
    mt.clip()
    mt.write(fname)
    print('Saved file "' + fname + '".')


def generate_music(model, midi, temperature, length=24*4*4, threshold=0.5, noise=False):
    """Generate some music!"""
    lookback = model.layers[0].input_shape[1]  # Changed as model is now no longer sequential.
    num_pitches = model.layers[0].input_shape[2]  # If sequential, remove [0] from both before [1] and [2]

    start_index = np.random.randint(0, midi.shape[0] - lookback + 1)  # Random starting spot
    seed = midi[start_index:start_index + lookback, :]
    seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))  # np.expanddim to orig

    print('\nGenerating roll with temp', temperature, 'at seed index of', start_index)
    sampled = seed.copy()

    output = np.zeros((1, length + lookback, num_pitches))
    output[0, :lookback, :] = seed

    # sampled = np.zeros((1, lookback, num_pitches))

    for i in range(length):
        preds = model.predict(sampled, verbose=0)
        extracted = extract_music(preds, temperature=temperature, threshold=threshold, noise=noise)
        # print('PREDS:', preds)
        # print(extracted)
        # print(lookback+i)

        output[0, lookback + i] = extracted  # Save the work
        sampled[:, :-1] = sampled[:, 1:]  # Move it over by one
        sampled[0, -1, :] = extracted  # Add it to the last row

    return output
