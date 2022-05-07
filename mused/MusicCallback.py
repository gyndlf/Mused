import tensorflow as tf
from .functions import generate_music, multi_save


class MusicCallback(tf.keras.callbacks.Callback):
    """Callback to generate music during training"""
    def __init__(self, gen_roll, gen_every=5, length=24*4*2, threshold=0.7, temp=0.5):
        super(MusicCallback, self).__init__()
        self.generated = {}
        self.gen_every = gen_every
        self.roll = gen_roll
        self.length = length
        self.threshold = threshold
        self.temp = temp

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.gen_every == 0:
            # Generate some music
            self.generated["epoch-" + str(epoch + 1)] = generate_music(self.model, self.roll, self.temp,
                                                                                length=self.length,
                                                                                threshold=self.threshold, noise=True)
            multi_save(self.generated, "out/generated/generated-during-training.mid")

    def on_train_end(self, logs=None):
        self.generated["last-generation"] = generate_music(self.model, self.roll, self.temp,
                                                                    length=self.length, threshold=self.threshold,
                                                                    noise=True)
        multi_save(self.generated, "out/generated/generated-during-training.mid")
