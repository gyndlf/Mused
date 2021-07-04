# d6595
# Heavy rewrite on d7042

# The code containing the actual functions for the midi. Can be used across multiple model architectures
# Basic midi <-> numpy interface
import matplotlib.pyplot as plt
import numpy as np
import pypianoroll
import os
import random
import keras

# TODO:
#  - Fix tempos. Make sure they are always the same
#  - Trim silence
#  - Force piece to be in the same key

MIDDLE_C = 64
MIDI_INPUTS = 128  # Length the rolls pitches must be


class Midi:
    """Is the controlling class for Midi. Can input multiple files to merge"""
    def __init__(self, num_pitches, beat_resolution=24, midi_dir='midi', cut=True):
        self.midi_dir = midi_dir
        self.beat_resolution = beat_resolution  # 24 has full 3/4 and 4/4 timings. 12 is ok. Is time-steps per quarter
        self.num_pitches = num_pitches  # Total number of pitches either side of middle c
        self.notes_above = num_pitches // 2
        self.cut = cut  # if to get rid of all pitches not in the range

        self.roll = None  # Is the pianoroll (Numpy array)

    def display(self, title='Piano Roll'):
        plt.figure(figsize=(8, 6))
        plt.matshow(self.roll, fignum=1, aspect='auto', cmap='plasma')
        plt.xlabel('Pitch')
        plt.ylabel('Time')
        plt.title(title)
        plt.show()

    def load_midi(self, fnames=None):
        if self.cut:
            print('Lower bound %s.' % (MIDDLE_C - self.notes_above))
            print('Upper bound %s.' % (MIDDLE_C + self.notes_above))
            print('Num pitches', self.num_pitches)

        rolls = []
        roll_length = 0
        for name in fnames:
            if name in os.listdir(self.midi_dir):
                multitrack = pypianoroll.read(os.path.join(self.midi_dir, name))
                multitrack.set_resolution(self.beat_resolution)
                # piano_multitrack.trim_trailing_silence()
                roll = multitrack.binarize().blend('any')
                print('---')
                print(name, 'input shape:', roll.shape)

                if self.cut:
                    # Adjust so that there are only the selected notes present
                    refined = roll[:, MIDDLE_C - self.notes_above:MIDDLE_C + self.notes_above]

                    loss = np.sum(roll) - np.sum(refined)
                    print('...Refined down', MIDI_INPUTS - self.notes_above*2, 'dimensions with', loss, 'note loss.')
                    print('...Loss of', (loss / np.sum(roll) * 100).__round__(2), '%')
                else:
                    refined = roll
                print('...Output shape:', refined.shape)

                rolls.append(refined)
                roll_length += refined.shape[0]

        # Merge all the rolls
        extended = np.zeros((roll_length, rolls[0].shape[1]), dtype='bool')  # Assuming that there is at least one roll
        print('Extended output shape', extended.shape)

        index = 0
        for roll in rolls:  # Fill in the empty roll
            extended[index:index + roll.shape[0], :] = roll
            index += roll.shape[0]

        self.roll = extended

    def load_np(self, roll):
        # Import pianoroll from numpy array
        if len(roll.shape) > 2:
            roll = np.reshape(roll, (roll.shape[1], roll.shape[2]))
        if roll.shape[1] > MIDI_INPUTS:
            print("ROLL ERROR: Too wide to fit into midi!")
        elif roll.shape[1] == MIDI_INPUTS:
            print("Correct roll width")
            self.cut = False
        else:
            self.num_pitches = roll.shape[1]
            self.notes_above = self.num_pitches // 2
            self.cut = True
            print("Roll is cut down to only", self.num_pitches, "notes")
        self.roll = roll

    def vectorise(self, lookback, step=1):
        # Now convert to phrases with a corresponding label
        phrases = []
        next_notes = []
        for i in range(0, self.roll.shape[0] - lookback, step):
            phrases.append(self.roll[i:i + lookback, :])  # Get the block
            next_notes.append(self.roll[i + lookback, :])  # The next line

        print(len(phrases), 'individual phrases.')

        # Vectorisation
        x = np.zeros((len(phrases), lookback, self.num_pitches), dtype='bool')
        y = np.zeros((len(phrases), self.num_pitches), dtype='bool')

        for i, phrase in enumerate(phrases):
            x[i, :, :] = phrase
            y[i, :] = next_notes[i]
        return x, y

    def reformat_roll(self):
        # Add back the original dimensions
        export = np.zeros((self.roll.shape[0], MIDI_INPUTS))
        export[:, MIDDLE_C-self.notes_above:MIDDLE_C+self.notes_above] = self.roll
        # export *= 100  # To make louder. Removed as bool
        return export

    def save(self, fname):
        if self.cut:
            roll = self.reformat_roll()
        else:
            roll = self.roll.copy()
        print('Output dim:', roll.shape)

        t = pypianoroll.BinaryTrack(pianoroll=roll, program=0, is_drum=False, name='Exported Pianoroll')
        mt = pypianoroll.Multitrack(tracks=[t])
        mt.clip()  # Make sure there aren't any crazy values
        print('Saving file "', fname, '".')
        mt.write(fname)

    def preview_data(self, midi_fname, fname='preview.mid', beat_resolution=None):
        # Test the functions to see if they work
        if beat_resolution is None:
            beat_resolution = self.beat_resolution
        print('Extracting "' + midi_fname + '" with beat resolution of', beat_resolution)
        self.cut = True
        self.load_midi([midi_fname])
        self.save(fname)


def extract_music(preds, temperature=1.0, threshold=None, noise=False):
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
    # Saves multiple tracks as one midi, for easy use within logic
    midi = Midi(70)
    tracks = []
    for i, temp in enumerate(piano_rolls):
        midi.load_np(piano_rolls[temp])
        roll = midi.reformat_roll()
        t = pypianoroll.BinaryTrack(pianoroll=roll, program=0, is_drum=False, name='Gen Roll T' + str(temp))
        tracks.append(t)
    print('Saving', len(tracks), 'tracks in one file "' + fname + '".')
    mt = pypianoroll.Multitrack(tracks=tracks)
    mt.clip()
    mt.write(fname)
    print('Saved file "' + fname + '".')


def generate_music(model, sample_roll, lookback, temperatures, num_pitches, length=24*4*4, threshold=None, noise=False):
    # Generate some music!
    start_index = random.randint(0, sample_roll.shape[0] - lookback - 1)  # Random starting spot
    print('Generating with seed index of', start_index)
    seed = sample_roll[start_index:start_index + lookback, :]
    seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))  # np.expanddim to orig

    generated = {}  # temperatures : output

    for temperature in temperatures:
        print('Generating roll with temp', temperature, 'and length', length)
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

        generated[temperature] = output
    return generated


def save_model(model, model_dir, fname):
    path = os.path.join(model_dir, fname)
    model.save(path)
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
