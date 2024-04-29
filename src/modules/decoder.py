import logging
import numpy as np

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


logger = logging.getLogger("Decoder")


class Decoder:

    model = None

    def __init__(
            self,
            height,
            width,
            depth,
            volume_size,
            filters,
            leaky_relu_alpha,
            strides,
            kernel,
            padding,
            activation,
            latent
    ):
        self.height = height
        self.width = width
        self.depth = depth
        self.volume_size = volume_size
        self.filters = filters
        self.leaky_relu_alpha = leaky_relu_alpha
        self.strides = strides
        self.kernel = kernel
        self.padding = padding
        self.latent = latent
        self.activation = activation

        # builds decoder model
        self.build_architecture()

    def build_architecture(self):

        # output of the encoder will be decoder inputs
        inputs = Input(shape=(self.latent,))
        x = Dense(np.prod(self.volume_size))(inputs)
        x = Reshape(self.volume_size)(x)

        # loop over reversed filters
        for f, k in zip(self.filters, self.kernel):
            x = Conv2DTranspose(f, k, strides=self.strides, padding=self.padding)(x)
            x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
            x = BatchNormalization(axis=-1)(x)

        # applies Conv2DTranspose layer to recover the original depth of the image
        x = Conv2DTranspose(self.depth, self.kernel[-1], padding=self.padding)(x)
        outputs = Activation(self.activation)(x)

        # build the decoder model
        self.model = Model(inputs, outputs, name="decoder")
    
    def predict(self, dataset):
        return self.model.predict(dataset)
