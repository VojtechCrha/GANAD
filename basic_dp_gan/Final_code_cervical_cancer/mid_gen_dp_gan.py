import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
# from tensorflow.python.checkpoint.checkpoint import Checkpoint
from tensorflow_privacy.privacy.optimizers import dp_optimizer as dp_opt
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from PIL import Image
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_squared_error
import os
from tqdm import tqdm
# from dataset_preparation_heartbeat import generate_heartbeat_dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from DPAE2 import DP_Autoencoder


# def load_mnist():
#     """Load and preprocess the MNIST dataset."""
#     (x_train, _), (_, _) = mnist.load_data()
#     x_train = x_train.astype('float32') / 255.0
#     x_train = np.expand_dims(x_train, axis=-1)
#     return x_train

# def load_heartbeat():
#     data = generate_heartbeat_dataset(100)
#     scaler = MinMaxScaler()
#     data = data

def make_generator_model(noise_dim):
    """Create the generator model to produce 179-column rows."""

    model = models.Sequential()

    model.add(layers.Dense(34, use_bias=False, input_shape=(noise_dim,)))

    # Increased dimensions to accommodate 179 columns
    # model.add(layers.Dense(16 * 12 * 256, use_bias=False, input_shape=(noise_dim,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # # Correct the reshape to (16, 15, 256) to align with the next layer
    # model.add(layers.Reshape((16, 15, 256)))
    #
    # # Use padding='same' to maintain spatial dimensions
    # model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # # Final layer to produce 179 columns with correct padding
    # model.add(layers.Conv2DTranspose(1, (6, 6), strides=(1, 1), padding='valid', use_bias=False, activation='sigmoid'))

    return model


def make_discriminator_model(noise_dim):
    """Create the discriminator model."""
    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=(noise_dim,)))  # Use a Dense layer for 1D input
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1))  # Output a single value for classification

    return model


def make_gan(generator, discriminator, noise_dim):
    """Create the GAN model."""
    discriminator.trainable = True
    gan_input = layers.Input(shape=(noise_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = models.Model(gan_input, gan_output)
    return gan


def dp_optimizer(epsilon, l2_norm_clip, noise_multiplier, batch_size):
    """Create and return a differentially private optimizer."""
    optimizer = dp_opt.DPAdamGaussianOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=batch_size,
        learning_rate=1e-4)
    # delta=delta)
    return optimizer


def train_step(generator, discriminator, gan, gen_optimizer, disc_optimizer, x_batch, batch_size, anomaly_detector,
               noise_dim):
    def gen_loss_fn():
        return gen_loss

    def disc_loss_fn():
        return disc_loss

    """Perform one training step with differential privacy."""
    noise = tf.random.normal([batch_size, noise_dim])
    # should I persist the gradients tape? Much worse performance but might be needed - tutorial does not use it
    with tf.GradientTape(persistent=False) as gen_tape, tf.GradientTape(persistent=False) as disc_tape:
        generated_images = generator(noise, training=True)

        # todo: implement mid-gen here

        original_train_indices = anomaly_detector.get_top_anomalies(x_batch, num_anomalies=1)
        clean_x_batch = np.delete(x_batch, original_train_indices, axis=0)

        fake_train_indices = anomaly_detector.get_top_anomalies(generated_images.numpy(), num_anomalies=1)
        clean_generated_images = np.delete(generated_images, fake_train_indices, axis=0)

        real_output = discriminator(clean_x_batch, training=True)
        fake_output = discriminator(clean_generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        # gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        gradients_of_generator = gen_optimizer.compute_gradients(gen_loss_fn, generator.trainable_variables,
                                                                 gradient_tape=gen_tape)
        gradients_of_discriminator = disc_optimizer.compute_gradients(disc_loss_fn, discriminator.trainable_variables,
                                                                      gradient_tape=disc_tape)

        # print(gradients_of_generator)
        # print(gradients_of_discriminator)

        gen_optimizer.apply_gradients(gradients_of_generator)
        disc_optimizer.apply_gradients(gradients_of_discriminator)


def generator_loss(fake_output):
    """Generator loss function."""
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.ones_like(fake_output))


def discriminator_loss(real_output, fake_output):
    """Discriminator loss function."""
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output, labels=tf.ones_like(real_output))
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output, labels=tf.zeros_like(fake_output))
    result = real_loss + fake_loss
    return result


def train_gan(generator, discriminator, gan, gen_optimizer, disc_optimizer, x_train, epochs, batch_size,
              data_samples_size, epsilon, anomaly_detector):
    """Train the GAN model."""

    # if use_checkpoint:
    #     checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for epoch in range(epochs * 100):  # should run infinitely (till the epsilon limit is reached), quick dirty fix
        print(f'Epoch: {epoch + 1}')
        steps_count = x_train.shape[0] // batch_size
        for step in tqdm(range(steps_count)):
            x_batch = x_train[step * batch_size:(step + 1) * batch_size]
            train_step(generator, discriminator, gan, gen_optimizer, disc_optimizer, x_batch, batch_size,
                       anomaly_detector,
                       noise_dim=x_batch.shape[1])
        # if (epochs % 1) == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        eps_old, _ = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy(x_train.shape[0], batch_size,
                                                                       gen_optimizer._noise_multiplier, (epoch + 1),
                                                                       delta=1e-5)
        # eps_new = compute_dp_sgd_privacy_lib.compute_dp_sgd_privacy_statement(x_train.shape[0], batch_size, (epoch + 1), gen_optimizer._noise_multiplier, delta=1e-5, used_microbatching=True)

        print(f"Epoch: {epoch + 1} finished, Epsilon old: {eps_old}")
        if eps_old > epsilon:
            print("Ended early due to epsilon budget being reached")
            break

    # Generate and save images after training
    # num_samples = 1000
    # synthetic_data = generate_synthetic_data(generator, num_samples, noise_dim)
    # save_generated_images(synthetic_data, epochs, step)


def generate_synthetic_data(generator, num_samples, noise_dim):
    """Generate synthetic data using the trained generator."""
    noise = tf.random.normal([num_samples, noise_dim])
    synthetic_data = generator(noise, training=False)
    return synthetic_data


def save_generated_images(images, epoch, step):
    """Save generated images as PNG files."""
    os.makedirs("generated_images", exist_ok=True)
    for i, image in enumerate(images):
        image = tf.squeeze(image, axis=-1)
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        tf.keras.preprocessing.image.save_img(f"generated_images/image_{epoch + 1}_{step + 1}_{i + 1}.png", image)


def save_generated_data_as_csv(data, filename):
    """Save generated data as CSV file."""
    flattened_data = np.reshape(data, (data.shape[0], -1))
    flattened_data = np.round(flattened_data, decimals=4)
    np.savetxt(filename, flattened_data, delimiter=",")


# After training_gan function
def save_models(generator, discriminator, epoch):
    """Save generator and discriminator models."""
    os.makedirs("saved_models", exist_ok=True)
    generator.save(f"saved_models/generator_epoch_{epoch}.keras")
    discriminator.save(f"saved_models/discriminator_epoch_{epoch}.keras")


def load_models(epoch):
    """Load generator and discriminator models."""
    generator = tf.keras.models.load_model(f"saved_models/generator_epoch_{epoch}.keras")
    discriminator = tf.keras.models.load_model(f"saved_models/discriminator_epoch_{epoch}.keras")
    return generator, discriminator


def dp_gan_main(args):
    # data
    # args.dataset == 'uci-epileptic':

    synth_data = []

    data = pd.read_csv('cervical-cancer_csv.csv')
    data = data.replace('?', np.nan)
    data.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], inplace=True, axis=1)
    numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
                    'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives (years)', 'IUD (years)',
                    'STDs (number)']
    categorical_df = ['Smokes', 'Hormonal Contraceptives', 'IUD',
                      'STDs', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
                      'STDs:vaginal condylomatosis',
                      'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                      'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
                      'STDs:molluscum contagiosum', 'STDs:AIDS',
                      'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV',
                      'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN',
                      'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology',
                      'Biopsy']
    # Fills the missing values of numeric data columns with mean of the column data.
    for feature in numerical_df:
        # print(feature, '', data[feature].apply(pd.to_numeric, errors='coerce').mean())
        feature_mean = round(data[feature].apply(pd.to_numeric, errors='coerce').mean(), 1)
        data[feature] = data[feature].fillna(feature_mean)
    for feature in categorical_df:
        data[feature] = data[feature].apply(pd.to_numeric,
                                            errors='coerce').fillna(1.0)

    data = MinMaxScaler().fit_transform(data)
    train_ratio = 0.5
    train = np.random.rand(data.shape[0]) < train_ratio
    train_data, test_data = data[train], data[~train]
    data_dim = data.shape[1]


    ae_budget = args.ad_budget * args.epsilon
    epsilon = args.epsilon * (1 - args.ad_budget)
    batch_size = args.batch_size
    delta = args.delta
    epochs = args.epochs
    noise_multiplier = args.lamda
    l2_norm_clip = 1.0
    data_samples_size = args.data_no
    # class_label = args['class_label']
    #
    # print("Testing on pre-gen anomaly-free data")
    anomaly_detector = DP_Autoencoder(data.shape[1], int(data.shape[1] / 2), epsilon=ae_budget)
    anomaly_detector.train(data)
    # print("Anomaly detector trained")

    # print(len(train_data))
    # original_train_indices = anomaly_detector.get_top_anomalies(train_data, num_anomalies=10)
    # clean_train_data = np.delete(train_data, original_train_indices, axis=0)

    for class_label in [0, 1]:
        train_class = train_data[train_data[:, -1] == class_label]

        generator = make_generator_model(data_dim)
        discriminator = make_discriminator_model(data_dim)
        gan = make_gan(generator, discriminator, data_dim)

        gen_optimizer = dp_optimizer(epsilon, l2_norm_clip, noise_multiplier, batch_size-1)
        disc_optimizer = dp_optimizer(epsilon, l2_norm_clip, noise_multiplier, batch_size-1)

        train_gan(generator, discriminator, gan, gen_optimizer, disc_optimizer, train_class, epochs, batch_size,
                  data_samples_size, epsilon, anomaly_detector)

        # synthetic_data = generate_synthetic_data(generator, num_samples, noise_dim)
        synth_data_class = generate_synthetic_data(generator, data_samples_size, data_dim).numpy()

        synth_data_class[:, -1] = class_label

        # print(synth_data_class.ndim)
        synth_data.append(synth_data_class)
        # synth_data = np.concatenate((synth_data, synth_data_class), axis=0)
    synth_data = np.concatenate(synth_data, axis=0)
    return synth_data, train_data, test_data
    # print(type(synth_data))
    # return synth_data

#
# if __name__ == "__main__":
#     # x_train = load_mnist()
#     noise_dim = 100
#     batch_size = 256
#     epochs = 500
#     epsilon = 1.0
#     use_checkpoint = False
#
#
#     # delta = 1e-5 # delta is currently not supported by the adam optimizer it seems
#
#     l2_norm_clip = 1.0
#     noise_multiplier = 1.1
#
#     generator = make_generator_model()
#     discriminator = make_discriminator_model()
#     gan = make_gan(generator, discriminator)
#
#     gen_optimizer = dp_optimizer(epsilon, l2_norm_clip, noise_multiplier, batch_size)
#     disc_optimizer = dp_optimizer(epsilon, l2_norm_clip, noise_multiplier, batch_size)
#
#     # checkpoint_dir = './training_checkpoints'
#     # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
#     # checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
#     #                                  discriminator_optimizier=disc_optimizer,
#     #                                  generator=generator,
#     #                                  discriminator=discriminator)
#
#     train_gan(generator, discriminator, gan, gen_optimizer, disc_optimizer, x_train, epochs, batch_size, use_checkpoint)
#     save_models(generator, discriminator, epochs)
#
#     num_samples = 1000
#     synthetic_data = generate_synthetic_data(generator, num_samples, noise_dim)
#
#
#     # Save synthetic data as CSV
#     save_generated_data_as_csv(synthetic_data, "synthetic_data.csv")
#
#     # Flatten the images for computing metrics
#     x_test_flat = x_train.reshape((x_train.shape[0], -1))
#     synthetic_data_flat = synthetic_data.numpy().reshape((synthetic_data.shape[0], -1))
