import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import time
from read_csv import *

# Configuration Parameters
char_set = "abcdefghijklmnopqrstuvwxyz0123456789-"
char_set_size = len(char_set)
domain_length = 10  # Maximum length of domain names
BUFFER_SIZE = 100000
BATCH_SIZE = 256
EPOCHS = 15
noise_dim = 100
num_examples_to_generate = 16

# Helper function to convert domain names to one-hot encoded format
def domain_to_one_hot(domains):
    one_hot = np.zeros((len(domains), domain_length, char_set_size), dtype=np.float32)
    for i, domain in enumerate(domains):
        for j, char in enumerate(domain):
            if j < domain_length:
                one_hot[i, j, char_set.index(char)] = 1.0
    return one_hot

# Placeholder for your domain name dataset loading logic
# For demonstration, we'll generate random domain names
def load_domain_dataset():
    domains = readdata()
    domains = remove_dot( domains )
    domains = domains[:BUFFER_SIZE]
    return domain_to_one_hot(domains)

train_domains = load_domain_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices(train_domains).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(256, use_bias=False, input_shape=(noise_dim,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Dense(512, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Dense(domain_length * char_set_size, activation='softmax'),
        layers.Reshape((domain_length, char_set_size))
    ])
    return model

def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(domain_length, char_set_size)),
        layers.Dense(512),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Dense(256),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Dense(1)
    ])
    return model

generator = make_generator_model()
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.optimizers.Adam(1e-4)
discriminator_optimizer = tf.optimizers.Adam(1e-4)

# Use this seed to visualize progress in the generated data
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(domains):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_domains = generator(noise, training=True)

        real_output = discriminator(domains, training=True)
        fake_output = discriminator(generated_domains, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for domain_batch in dataset:
            train_step(domain_batch)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# 5.8s per epoch on laptop
train(train_dataset, EPOCHS)

def one_hot_to_domain(one_hot_domains):
    # Convert one-hot encoded domains back to string domain names
    domain_names = []
    for domain in one_hot_domains:
        # Convert the TensorFlow tensor to a NumPy array before using argmax
        domain_array = domain.numpy()
        char_indices = domain_array.argmax(axis=-1)  # Now we can use argmax on a NumPy array
        domain_name = ''.join(char_set[index] for index in char_indices)
        domain_names.append(domain_name.rstrip('-'))  # Remove padding if any
    return domain_names


def generate_domain_names(model, num_samples):
    # Generate random noise
    noise = tf.random.normal([num_samples, noise_dim])
    # Use the generator to create domain names from the noise
    generated_domains_one_hot = model(noise, training=False)
    # Convert the one-hot encoded domain names to string domain names
    domain_names = one_hot_to_domain(generated_domains_one_hot)
    return domain_names

# Now, let's generate some domain names
generated_domain_names = generate_domain_names(generator, 10)  # Generate 10 domain names
for domain_name in generated_domain_names:
    print(domain_name)
