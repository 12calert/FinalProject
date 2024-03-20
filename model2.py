from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense, Reshape, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from read_csv import *

class GAN:
    def __init__(self, input_dim, discriminator_conv_filters, discriminator_conv_kernel_size, discriminator_conv_strides, discriminator_batch_norm_momentum, discriminator_activation, discriminator_dropout_rate, discriminator_learning_rate, generator_initial_dense_layer_size, generator_upsample, generator_conv_filters, generator_conv_kernel_size, generator_conv_strides, generator_batch_norm_momentum, generator_activation, generator_dropout_rate, generator_learning_rate, optimiser, z_dim):
        self.input_dim = input_dim
        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.n_layers_generator = len(generator_conv_filters)
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        self.optimiser = optimiser
        self.z_dim = z_dim

    def build_discriminator(self):
        discriminator_input = Input(shape=self.input_dim, name='discriminator_input')
        x = discriminator_input
        for i in range(self.n_layers_discriminator):
            x = Conv2D(
                filters=self.discriminator_conv_filters[i],
                kernel_size=self.discriminator_conv_kernel_size[i],
                strides=self.discriminator_conv_strides[i],
                padding='same',
                name='discriminator_conv_' + str(i)
            )(x)

            if self.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum=self.discriminator_batch_norm_momentum)(x)
                x = Activation(self.discriminator_activation)(x)
                if self.discriminator_dropout_rate:
                    x = Dropout(rate=self.discriminator_dropout_rate)(x)
        x = Flatten()(x)
        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer='glorot_normal')(x)
        discriminator = Model(discriminator_input, discriminator_output)
        return discriminator

    def build_generator(self):
        generator_input = Input(shape=(self.z_dim,), name='generator_input')
        x = generator_input
        x = Dense(np.prod(self.generator_initial_dense_layer_size))(x)
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
        x = Activation(self.generator_activation)(x)
        x = Reshape(self.generator_initial_dense_layer_size)(x)
        if self.generator_dropout_rate:
            x = Dropout(rate=self.generator_dropout_rate)(x)
        for i in range(self.n_layers_generator):
            x = UpSampling2D()(x)
            x = Conv2D(
                filters=self.generator_conv_filters[i],
                kernel_size=self.generator_conv_kernel_size[i],
                strides=self.generator_conv_strides[i],
                padding='same',
                name='generator_conv_' + str(i)
            )(x)
            if i < self.n_layers_generator - 1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum=self.generator_batch_norm_momentum)(x)
                x = Activation('relu')(x)
            else:
                x = Activation('tanh')(x)
        generator_output = x
        generator = Model(generator_input, generator_output)
        return generator

    def compile_models(self):
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            optimizer=RMSprop(lr=self.discriminator_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.discriminator.trainable = False
        self.generator = self.build_generator()

        model_input = Input(shape=(self.z_dim,), name='model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input, model_output)

        self.model.compile(
            optimizer=RMSprop(lr=self.generator_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def train_discriminator(self, x_train, batch_size):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        true_imgs = x_train[idx]
        self.discriminator.train_on_batch(true_imgs, valid)
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        self.discriminator.train_on_batch(gen_imgs, fake)

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        self.model.train_on_batch(noise, valid)

    def train(self, x_train, epochs=2000, batch_size=64):
        for epoch in range(epochs):
            self.train_discriminator(x_train, batch_size)
            self.train_generator(batch_size)

gan = GAN(input_dim=(28, 28, 1),
          discriminator_conv_filters=[64, 64, 128, 128],
          discriminator_conv_kernel_size=[5, 5, 5, 5],
          discriminator_conv_strides=[2, 2, 2, 1],
          discriminator_batch_norm_momentum=None,
          discriminator_activation='relu',
          discriminator_dropout_rate=0.4,
          discriminator_learning_rate=0.0008,
          generator_initial_dense_layer_size=(7, 7, 64),
          generator_upsample=[2, 2, 1, 1],
          generator_conv_filters=[128, 64, 64, 1],
          generator_conv_kernel_size=[5, 5, 5, 5],
          generator_conv_strides=[1, 1, 1, 1],
          generator_batch_norm_momentum=0.9,
          generator_activation='relu',
          generator_dropout_rate=None,
          generator_learning_rate=0.0004,
          optimiser='rmsprop',
          z_dim=100)

gan.compile_models()

import tensorflow as tf

# Load MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
x_train = np.expand_dims(x_train, axis=-1)   # Add channel dimension for grayscale images

# Print the shape of the training data
print("Training data shape:", x_train.shape)

# Train the GAN
gan.train(x_train, epochs=2000, batch_size=64)
