# d7043

# Generate music based on a seed

import argparse
import functions
import numpy as np
import pypianoroll
import train


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
    midi = functions.Midi(70)
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


def generate_music(model, midi, temperatures, length=24*4*4, threshold=None, noise=False):
    # Generate some music!
    lookback = model.layers[0].input_shape[1]
    num_pitches = model.layers[0].input_shape[2]

    start_index = np.random.randint(0, midi.shape[0] - lookback - 1)  # Random starting spot
    print('Generating with seed index of', start_index)
    seed = midi[start_index:start_index + lookback, :]
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
            extracted = extract_music(preds, temperature=temperature, threshold=0.5, noise=noise)
            # print('PREDS:', preds)
            # print(extracted)
            # print(lookback+i)

            output[0, lookback + i] = extracted  # Save the work
            sampled[:, :-1] = sampled[:, 1:]  # Move it over by one
            sampled[0, -1, :] = extracted  # Add it to the last row

        generated[temperature] = output
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate midi according to the sample input and model.")
    parser.add_argument(
        'midi', type=str, help="Input midi file to sample")
    parser.add_argument(
        'model', type=str, help="Input model to generate with")
    parser.add_argument(
        '-l', '--length', required=False, type=int, help="Length of generated string. Default [24*4*4]")
    parser.add_argument(
        '-t', '--temp', required=False, nargs='+', type=float, help="Temperatures to bake with. Default [0.6]")
    parser.add_argument(
        '-n', '--noise', action='store_true', help="Add noise to the result. Else default to threshold"
    )
    args = parser.parse_args()

    if args.length is None:
        args.length = 24*4*4  # Set it to 4 bars

    if args.temp is None:
        args.temp = [0.6]

    model = train.load_model(args.model)
    num_pitches = model.layers[0].input_shape[2]

    roller = functions.Midi(num_pitches)
    roller.load_midi([args.midi])

    print(generate_music(model, roller.roll, args.temp, length=args.length, noise=args.noise))


if __name__ == '__main__':
    main()
