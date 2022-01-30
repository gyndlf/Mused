import numpy as np
import pypianoroll
import matplotlib.pyplot as plt
import sys


MIDDLE_C = 64
MIDI_INPUTS = 128  # Length the rolls pitches must be
DRUM_NAMES = ["drums", "congos", "cymbals", "hits"]
PIANO_NAMES = ["piano", "accoustic piano", 'melody  (bb)']
BASS_NAMES = ["bass", "fretless bass"]


class Midi: 
    """Is the controlling class for Midi. Can input multiple files to merge"""
    def __init__(self, num_pitches, beat_resolution=24, cut=True):
        self.beat_resolution = beat_resolution  # 24 has full 3/4 and 4/4 timings. 12 is ok. Is time-steps per quarter
        self.num_pitches = num_pitches  # Total number of pitches either side of middle c
        self.notes_above = num_pitches // 2
        self.cut = cut  # if to get rid of all pitches not in the range
        self.tempo = None  # Approx tempo of the piece
        self.roll = None  # Is the pianoroll (Numpy array)

    def display(self, title='Piano Roll'):
        """Display the piano roll as a plot"""
        plt.figure(figsize=(8, 6))
        plt.matshow(self.roll, fignum=1, aspect='auto', cmap='plasma')
        plt.xlabel('Pitch')
        plt.ylabel('Time')
        plt.title(title)
        plt.show()

    def is_piano(self, track: pypianoroll.Track) -> bool:
        return track.name.lower().encode("ascii", errors="ignore") in PIANO_NAMES


    def is_drum(self, track: pypianoroll.Track) -> bool:
        return track.is_drum or (track.name.lower().encode("ascii", errors="ignore") in DRUM_NAMES)


    def is_bass(self, track: pypianoroll.Track) -> bool:
        return track.name.lower().encode("ascii", errors="ignore") in BASS_NAMES

    def load_midi(self, fnames):
        """Load the midi, process it and save"""
        if self.cut:
            print('Lower bound %s.' % (MIDDLE_C - self.notes_above))
            print('Upper bound %s.' % (MIDDLE_C + self.notes_above))
            print('Num pitches', self.num_pitches)

        rolls = []
        roll_length = 0
        for fname in fnames:
            multitrack = pypianoroll.read(fname)
            multitrack.set_resolution(self.beat_resolution)
            self.tempo = multitrack.tempo.mean()
            print('---')
            print("Loading", fname)

            analysis = {"piano":0, "drums":0, "bass":0}
            for track in multitrack.tracks:
                if not (self.is_bass(track) or self.is_piano(track) or self.is_drum(track)):
                    print("WARNING: Unable to classify track:", track.name.lower())
                    print("Please add the name to the list in Midi.py so this does not happen again!!")
                    sys.exit(-1)
                elif self.is_piano(track):
                    analysis["piano"] += 1
                elif self.is_bass(track):
                    analysis["bass"] += 1
                elif self.is_drum(track):
                    analysis["drums"] += 1

            print("...Track analysis of", analysis)

            multitrack.tracks[:] = [x for x in multitrack.tracks if (not (self.is_drum(x) or self.is_bass(x))) and self.is_piano(x)]

            # piano_multitrack.trim_trailing_silence()
            roll = multitrack.binarize().blend('any')
            print('...Input shape:', roll.shape)

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
        extended = np.zeros((roll_length, self.num_pitches), dtype='bool')  # Assuming that there is at least one roll
        print('Extended output shape', extended.shape)

        index = 0
        for roll in rolls:  # Fill in the empty roll
            extended[index:index + roll.shape[0], :] = roll
            index += roll.shape[0]

        self.roll = extended

    def load_np(self, roll, tempo=120):
        """Import pianoroll from numpy array"""
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
        self.tempo = tempo

    def vectorise(self, lookback, step=1):
        """Convert to phrases with a corresponding label"""
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
        """Add back the original dimensions"""
        export = np.zeros((self.roll.shape[0], MIDI_INPUTS))
        export[:, MIDDLE_C-self.notes_above:MIDDLE_C+self.notes_above] = self.roll
        # export *= 100  # To make louder. Removed as bool
        return export

    def save(self, fname):
        """Save the midi as .midi"""
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

    def preview_data(self, midi_fname, fname='out/generated/preview.mid', beat_resolution=None):
        """Test the functions to see if they work"""
        if beat_resolution is None:
            beat_resolution = self.beat_resolution
        print('Extracting "' + midi_fname + '" with beat resolution of', beat_resolution)
        self.cut = True
        self.load_midi([midi_fname])
        self.save(fname)


if __name__ == "__main__":
    m = Midi(60, beat_resolution=24)
    m.preview_data("../resources/jazz/Caravan2.mid", fname="../out/generated/preview.mid")
