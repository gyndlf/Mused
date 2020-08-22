# d6595
# The code containing the actual functions for the midi. Can be used across multiple model architectures
# Basic midi <-> numpy interface
import matplotlib.pyplot as plt
import numpy as np
from pypianoroll import Track, Multitrack
import pypianoroll as pr
import os
import random

class Midi():
    def __init__(self, middle_c=64, midi_inputs=128, num_notes_below_c=30, num_notes_above_c=30, beat_resolution=12):
        self.MIDDLE_C = middle_c
        self.MIDI_INPUTS = midi_inputs
        self.num_notes_below_c = num_notes_below_c
        self.num_notes_above_c = num_notes_above_c
        self.beat_resolution = beat_resolution
        self.num_pitches = self.num_notes_above_c + self.num_notes_below_c
        self.bar_length = self.beat_resolution * 4 * 2

    def display_pianoroll(self, roll, title='Piano Roll'):
        plt.figure(figsize=(8, 6))
        plt.matshow(roll, fignum=1, aspect='auto', cmap='plasma')
        plt.xlabel('Pitch')
        plt.ylabel('Time')
        plt.title(title)
        plt.show()

    def plot_values(self, history):
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

    def load_midi(self, midi_dir, num_files=None, over_ride_fnames=None, display=False, cut=True):
        fnames = os.listdir(midi_dir)
        print('Found', len(fnames), 'files:', fnames)

        if cut:
            print('Lower bound %s.' % (self.MIDDLE_C - self.num_notes_below_c))
            print('Upper bound %s.' % (self.MIDDLE_C + self.num_notes_above_c))
            print('Num pitches', self.num_pitches)

        data = []
        total_length = 0
        if num_files is not None:  # Only get specified length
            fnames = fnames[:num_files]

        if over_ride_fnames is not None:
            fnames = over_ride_fnames

        print('Refined midi files', fnames)

        for fname in fnames:
            file = os.path.join(midi_dir, fname)
            piano_multitrack = pr.Multitrack(file, beat_resolution=self.beat_resolution)

            piano_multitrack.trim_trailing_silence()
            piano_multitrack.binarize()

            pianoroll = piano_multitrack.get_merged_pianoroll()

            if display:
                self.display_pianoroll(pianoroll, title='RAW')

            print(fname, 'input shape:', pianoroll.shape)

            if cut:
                # Adjust so that there are only the selected notes present
                refined = pianoroll[:, self.MIDDLE_C - self.num_notes_below_c:self.MIDDLE_C + self.num_notes_above_c]
            else:
                refined = pianoroll

            if display:
                self.display_pianoroll(refined, title='Refined')

            if cut:
                loss = np.sum(pianoroll) - np.sum(refined)
                print('...Refined down', 128 - self.num_notes_above_c - self.num_notes_below_c, 'dimensions with', loss, 'note loss.')
                print('...Loss of', (loss / np.sum(pianoroll) * 100).__round__(2), '%')
            print('...Output shape:', refined.shape)

            data.append(refined)
            total_length += refined.shape[0]

        extended = np.zeros((total_length, data[0].shape[1]), dtype='bool')
        print('extended output shape', extended.shape)

        index = 0
        for sequence in data:
            extended[index:index + sequence.shape[0], :] = sequence
            index += sequence.shape[0]
        if display:
            self.display_pianoroll(extended, title='Extended shape')
        return extended

    def vectorise(self, roll, lookback, step=1):
        # Now convert to phrases with a corresponding label
        phrases = []
        next_notes = []
        for i in range(0, roll.shape[0] - lookback, step):
            phrases.append(roll[i:i + lookback, :])  # Get the block
            next_notes.append(roll[i + lookback, :])  # The next line

        print(len(phrases), 'individual phrases.')

        # Vectorisation
        x = np.zeros((len(phrases), lookback, self.num_pitches), dtype='bool')
        y = np.zeros((len(phrases), self.num_pitches), dtype='bool')

        for i, phrase in enumerate(phrases):
            x[i, :, :] = phrase
            y[i, :] = next_notes[i]
        return x, y

    def extract_music(self, preds, temperature=1.0, threshold=None):
        if threshold is not None:
            print('Centering around threshold of ' + str(threshold) +'.')
            # Adjust the music to be 1 if above the threshold, else 0
            split = preds - threshold + 0.5  # Center around 0.5
            split = np.round(split, 0)  # return np.round(split, 0)
            return split
        preds = np.asarray(preds).astype('float64')  # Convert to np array
        preds = np.log(preds) / temperature
        preds = np.exp(preds)
        r = np.random.random(preds.shape)
        output = np.zeros(preds.shape)
        output[r < preds] = 1  # has to be below the critera
        return output
        # p/p.sum(axis=0) (Normalise)

    def process_roll(self, x, display=False):
        # Add original dimensions
        x = np.reshape(x, (x.shape[1], x.shape[2]))  # np.expand_dims(l2, axis)
        #if display:
        #    self.display_pianoroll(x, title='Original Dimensions')
        y = np.zeros((x.shape[0], self.MIDI_INPUTS))
        y[:, self.MIDDLE_C - self.num_notes_below_c:self.MIDDLE_C + self.num_notes_above_c] = x
        if display:
            self.display_pianoroll(y, title='Resized Dimensions')
        y *= 100  # Make it a bit louder
        return y

    def generate(self, model, sample_roll, lookback, temperatures, length=16, threshold=None):
        # Generate some music!
        start_index = random.randint(0, sample_roll.shape[0] - lookback - 1)  # Random starting spot
        print('Generating with seed index of', start_index)
        seed = sample_roll[start_index:start_index + lookback, :]
        seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))  # np.expanddim to orig

        generated_length = self.bar_length * length

        generated = {}  # temperatures : output

        for temperature in temperatures:
            print('Generating roll with temp', temperature, 'and length', generated_length)
            sampled = seed.copy()

            output = np.zeros((1, generated_length + lookback, self.num_pitches))
            output[0, :lookback, :] = seed

            # sampled = np.zeros((1, lookback, num_pitches))

            for i in range(generated_length):
                preds = model.predict(sampled, verbose=0)
                extracted = self.extract_music(preds, temperature=temperature, threshold=threshold)
                # print('PREDS:', preds)
                # print(extracted)
                # print(lookback+i)

                output[0, lookback + i] = extracted  # Save the work
                sampled[:, :-1] = sampled[:, 1:]  # Move it over by one
                sampled[0, -1, :] = extracted  # Add it to the last row

            generated[temperature] = output
        return generated

    def save(self, piano_roll, fname='test.mid', display=True, process=True):
        roll = piano_roll.copy()
        #print('Input dim:', roll.shape)
        # test = test[:, lookback:, :]

        if process:
            roll = self.process_roll(roll, display=display)
            #print('Output dim:', roll.shape)

        t = Track(pianoroll=roll, program=0, is_drum=False, name='ai gen pianoroll')
        mt = Multitrack(tracks=[t], tempo=(120.0 * (self.beat_resolution / 24)).__round__(1))  # Percent speed up
        print('Saving file "', fname, '".')
        mt.write(fname)

    def save_model(self, model, model_dir, fname='model.h5'):
        path = os.path.join(model_dir, fname)
        model.save(path)
        print('Saved model "' + fname + '"')

    def load_model(self, models, model_dir, fname):
        path = os.path.join(model_dir, fname)
        model = models.load_model(path)
        print('Loaded model "' + fname + '"')
        return model

    def smart_save(self, piano_rolls, fname='generated.mid', display=True):
        # Saves multiple tracks as one midi, for easy use within logic
        tracks = []
        for i, temp in enumerate(piano_rolls):
            roll = self.process_roll(piano_rolls[temp], display=display)
            t = Track(pianoroll=roll, program=0, is_drum=False, name='Gen Roll T' + str(temp))
            tracks.append(t)
        print('Saving', len(tracks), 'tracks in one file "' + fname + '".')
        mt = Multitrack(tracks=tracks, tempo=(120.0 * (self.beat_resolution / 24)).__round__(2))
        mt.write(fname)
        print('Saved file "' + fname + '".')

    def preview_data(self, midi_dir, midi_file, fname='preview.mid', beat_resolution=None):
        # Extracts the right music with the beat_resolution factor
        if beat_resolution is None:
            beat_resolution = self.beat_resolution
        print('Extracting "' + midi_file + '" with beat resolution of', beat_resolution)
        pianoman = self.load_midi(midi_dir, over_ride_fnames=[midi_file], cut=False)
        self.save(pianoman, fname=fname, process=False)


