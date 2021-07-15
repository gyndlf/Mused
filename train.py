# d7043

# Train the specified model with input Midi
# Training model functions

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from time import time
import tensorflow as tf
from tensorflow import keras  # import from tensorflow for better support??? I dunno
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt
import generate


class GenerateMusic(tf.keras.callbacks.Callback):
    """Callback to generate music during training"""
    def __init__(self, gen_roll, gen_every=5, length=24*4*2, threshold=0.7, temp=0.5):
        super(GenerateMusic, self).__init__()
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


class Gru:
    """The controlling class for the training model"""
    def __init__(self, name, model_dir="models/"):
        self.model = None
        self.name = name
        self.model_dir = model_dir

    def build(self, lookback, num_pitches, loss='binary_crossentropy'):
        # Build the model architecture
        #model = models.Sequential()
        #model.add(layers.LSTM(256, input_shape=(lookback, num_pitches), return_sequences=True,
        #                      dropout=0.3, recurrent_dropout=0.3))
        #model.add(layers.LSTM(512, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
        #model.add(layers.LSTM(256, return_sequences=False))
        #model.add(layers.Dense(256, activation='relu'))
        #model.add(layers.Dropout(0.3))
        # model.add(layers.Dense(64, activation='relu'))
        #model.add(layers.Dense(num_pitches, activation='sigmoid'))

        input_layer = layers.Input(shape=(lookback, num_pitches))
        conv1 = layers.Conv1D(filters=32, kernel_size=8, strides=1, activation='relu', padding='same')(input_layer)
        lstm1 = layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,)(conv1)
        lstm2 = layers.LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,)(lstm1)
        lstm3 = layers.LSTM(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,)(lstm2)
        dense1 = layers.Dense(256, activation='relu')(lstm3)
        dropout1 = layers.Dropout(0.3)(dense1)
        output_layer = layers.Dense(num_pitches, activation='sigmoid')(dropout1)
        model = models.Model(inputs=input_layer, outputs=output_layer)

        model.summary()
        print("Model built.")

        model.compile(loss=loss,  # categorical_crossentropy or mse or binary_crossentropy
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=["accuracy", "mean_absolute_error"])
        self.model = model

    def set_model(self, model):
        print("Overwriting model -- Ensure that correct dimensions are being used")
        self.model = model

    def load(self, fname):
        self.model = load_model(fname)

    def save(self, fname):
        save_model(self.model, fname)

    def train(self, x, y, epochs, callbacks, batch_size=128):
        tic = time()
        history = self.model.fit(x, y,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=1,
                                 callbacks=callbacks)

        historys = [history]  # Kept for legacy purposes
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


def main():
    # Train the network from the command line
    parser = argparse.ArgumentParser(description="Train the model using the input settings.")
    parser.add_argument(
        'midi', type=str, nargs='+', help="Input midi file to train from")
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

    gru = Gru(args.name)
    gru.build(args.lookback, args.num_notes, loss=args.loss)

    if args.model is not None:
        print("Loading model " + args.model + " to over-ride weights")
        gru.model = load_model(args.model)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=args.patience, restore_best_weights=False, monitor="loss"),
        # Stop early if training is only going ok
        GenerateMusic(music.roll, gen_every=args.gen_every, temp=args.generate_temp),  # Generate some music
        tf.keras.callbacks.ModelCheckpoint(gru.model_dir + gru.name + ".h5",
                                           save_best_only=True, monitor="loss"),
        tf.keras.callbacks.ProgbarLogger(),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    print("Training model with %s settings" % args)
    print("Change model architecture in train.py")

    historys = gru.train(x, y, args.epochs, callbacks, batch_size=args.batch)
    #plot_history(historys)


if __name__ == '__main__':
    import argparse
    import functions
    import datetime
    main()