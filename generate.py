# d7043
#
# Generate music based on a seed
# Generation functions:
#
# IS IRRESPECTIVE OF IF THE MODEL IS GRU OR VECTOR. JUST LOADS AND GOES

import mused

# TODO:
#  - Generate music in parallel


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate midi according to the sample input and model.")
    parser.add_argument(
        'midi', type=str, help="Input midi file to sample")
    parser.add_argument(
        'model', type=str, help="Input model to generate with")
    parser.add_argument(
        '-l', '--length', required=False, type=int, default=24*4*4, help="Length of generated string. Default [24*4*4]")
    parser.add_argument(
        '-t', '--temp', required=False, default=[0.2, 0.4, 0.45, 0.6, 0.8, 1.0], nargs='+', type=float, help="Temperatures to bake with. Default [range]")
    parser.add_argument(
        '--no-noise', action='store_false', help="Remove noise to the result. Default to false"
    )
    parser.add_argument(
        '--thresh', default=0.5, type=float, help="Set threshold value. Default [0.5]"
    )
    args = parser.parse_args()

    print("Using settings (Remember, no-noise is opposite)", args)
    model = mused.functions.load_model(args.model)
    model.summary()
    num_pitches = model.layers[0].input_shape[2]  # if not sequential add [0] before [2]

    roller = mused.Midi(num_pitches)
    roller.load_midi([args.midi])

    print("Using length of %s and threshold of %s or potentially with noise (%s)" %
          (args.length, args.thresh, args.no_noise))

    outputs = {}
    for temp in args.temp:
        outputs["Temp-" + str(temp)] = mused.generate_music(model, roller.roll, temp, length=args.length,
                                                      noise=args.no_noise, threshold=args.thresh)
    mused.multi_save(outputs, 'out/generated/generated.mid')


if __name__ == '__main__':
    main()
