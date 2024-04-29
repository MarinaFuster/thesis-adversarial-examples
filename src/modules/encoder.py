import logging

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


logger = logging.getLogger("Encoder")


class Encoder:

    def __init__(self, height, width, depth, filters, leaky_relu_alpha, strides, kernel, padding, latent):
        self.input_shape = (height, width, depth)
        self.filters = filters
        self.leaky_relu_alpha = leaky_relu_alpha
        self.strides = strides
        self.kernel = kernel
        self.padding = padding
        self.latent = latent

        self.inputs = None
        self.volume_size = None
        self.model = None

        # builds encoder model
        self.build_architecture()
    
    def build_architecture(self):

        self.inputs = Input(shape=self.input_shape)
        x = self.inputs

        # loop over filters
        for f, k in zip(self.filters, self.kernel):
            x = Conv2D(f, k, strides=self.strides, padding=self.padding)(x)
            x = LeakyReLU(alpha=self.leaky_relu_alpha)(x)
            x = BatchNormalization(axis=-1)(x)

        # flattens resulting image
        self.volume_size = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(self.latent)(x)
        latent_layer = LeakyReLU(alpha=self.leaky_relu_alpha)(x)

        # build the encoder model
        self.model = Model(self.inputs, latent_layer, name="encoder")
    
    def predict(self, dataset):
        return self.model.predict(dataset)
