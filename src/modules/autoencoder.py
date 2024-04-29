import logging

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from modules.decoder import Decoder
from modules.encoder import Encoder

logger = logging.getLogger("Autoencoder")


class Autoencoder:

    def __init__(self):
        self.height = 256
        self.width = 256
        self.depth = 3
        self.filters = [16, 32, 64]
        self.leaky_relu_alpha = 0.2
        self.adam_learning_rate = 0.001
        self.strides = 2
        self.kernel = [(3, 3), (5, 5), (7, 7)]
        self.padding = 'same'
        self.decoder_activation = 'sigmoid'
        self.latent = 64
        self.encoder = None
        self.decoder = None
        self.model = None
        self.H = None

    def set_height(self, height):
        self.height = height
        return self

    def set_width(self, width):
        self.width = width
        return self

    def set_depth(self, depth):
        self.depth = depth
        return self

    def set_filters(self, filters):
        self.filters = filters
        return self

    def set_leaky_relu_alpha(self, leaky_relu_alpha):
        self.leaky_relu_alpha = leaky_relu_alpha
        return self

    def set_adam_learning_rate(self, adam_learning_rate):
        self.adam_learning_rate = adam_learning_rate
        return self

    def set_strides(self, strides):
        self.strides = strides
        return self

    def set_kernel(self, kernel):
        self.kernel = kernel
        return self

    def set_padding(self, padding):
        self.padding = padding
        return self

    def set_decoder_activation(self, decoder_activation):
        self.decoder_activation = decoder_activation
        return self

    def set_latent(self,latent):
        self.latent = latent
        return self

    # must be called in order to train the autoencoder
    def build_model(self):
        encoder = Encoder(self.height, self.width, self.depth,
                          self.filters, self.leaky_relu_alpha, self.strides,
                          self.kernel, self.padding, self.latent)
        decoder = Decoder(self.height, self.width, self.depth, encoder.volume_size[1:],
                          self.filters[::-1], self.leaky_relu_alpha, self.strides,
                          self.kernel[::-1], self.padding, self.decoder_activation, self.latent)

        inputs = encoder.inputs

        self.encoder = encoder.model
        self.decoder = decoder.model
        self.model = Model(inputs, self.decoder(self.encoder(inputs)), name="autoencoder")

        optimizer = Adam(lr=self.adam_learning_rate)
        self.model.compile(loss="mse", optimizer=optimizer)

        return self

    # method must be called after build()
    def train(self, epochs, batch_size, train_dataset, test_dataset, tensorboard_callback=None):
        callbacks = None
        if tensorboard_callback:
            callbacks = [tensorboard_callback]

        # training history
        self.H = self.model.fit(
            train_dataset, train_dataset,
            validation_data=(test_dataset, test_dataset),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    def predict(self, dataset):
        return self.model.predict(dataset)
