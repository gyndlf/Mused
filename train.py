# d7043

# Train the specified model with input Midi
# Training model functions

import tensorflow as tf
import generate
import Gru
import Vector


class MakeMusic(tf.keras.callbacks.Callback):
    """Callback to generate music during training"""
    def __init__(self, gen_roll, gen_every=5, length=24*4*2, threshold=0.7, temp=0.5):
        super(MakeMusic, self).__init__()
        self.generated = {}
        self.gen_every = gen_every
        self.roll = gen_roll
        self.length = length
        self.threshold = threshold
        self.temp = temp

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.gen_every == 0:
            # Generate some music
            self.generated["epoch-" + str(epoch + 1)] = generate.generate_music(self.model, self.roll, self.temp,
                                                                                length=self.length,
                                                                                threshold=self.threshold, noise=True)

    def on_train_end(self, logs=None):
        self.generated["last-generation"] = generate.generate_music(self.model, self.roll, self.temp,
                                                                    length=self.length, threshold=self.threshold,
                                                                    noise=True)
        generate.multi_save(self.generated, "outputs/generated-during-training.mid")


def main():
    # Train the network from the command line
    parser = argparse.ArgumentParser(description="Train the model using the input settings.")
    parser.add_argument(
        'midi', type=str, nargs='+', help="Input midi file to train from. '+' for all")
    parser.add_argument(
        '-r', '--resolution', type=int, required=False, default=24, help="Beat resolution of a quarter note (1/4) [24]")
    parser.add_argument(
        "-l", '--loss', type=str, required=False, default="binary_crossentropy", help="Loss function to use [binary_crossentropy]")
    parser.add_argument(
        'name', type=str, help="Name for the model. Usually 'lstm-v?'")
    parser.add_argument(
        "epochs", type=int, help="Number of epochs to train for")
    parser.add_argument(
        "--patience", type=int, required=False, default=3, help="How long to wait [3]")
    parser.add_argument(
        "--generate-temp", type=float, required=False, default=0.55, help="Temperature during generation while training [0.55]")
    parser.add_argument(
        "-g", "--gen-every", type=int, required=False, default=3, help="Generate music every [3] epochs")
    parser.add_argument(
        "-n", "--num-notes", type=int, required=False, default=50, help="Total range of notes to work with [50]")
    parser.add_argument(
        "-lb", "--lookback", type=int, required=False, default=24*4*2, help="How far the model looks back [24*4*4]")
    parser.add_argument(
        "-b", "--batch", type=int, required=False, default=128, help="Batch size [128]")
    parser.add_argument(
        "--model", type=str, required=False, default=None, help="Model to continue to train from [None]")

    args = parser.parse_args()

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    music = functions.Midi(args.num_notes, beat_resolution=args.resolution)
    music.load_midi(args.midi)
    #music.display()

    x, y = music.vectorise(args.lookback, step=1)
    tempo = music.tempo

    gru = Gru.Gru(args.name)
    gru.build(args.lookback, args.num_notes, loss=args.loss)

    if args.model is not None:
        print("Loading model " + args.model + " to over-ride weights")
        gru.model = functions.load_model(args.model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=False, monitor="loss"),
        # Stop early if training is only going ok
        MakeMusic(music.roll, gen_every=args.gen_every, temp=args.generate_temp),  # Generate some music
        tf.keras.callbacks.ModelCheckpoint(gru.model_dir + gru.name + ".h5",
                                           save_best_only=True, monitor="loss"),
        tf.keras.callbacks.ProgbarLogger(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    print("Training model with %s settings" % args)
    print("Change model architecture in train.py")

    historys = gru.train(x, y, args.epochs, callbacks, batch_size=args.batch)
    #functions.plot_history(historys)


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import argparse
    import functions
    import datetime
    main()
