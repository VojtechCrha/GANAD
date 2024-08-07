import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DP_Autoencoder:
    def __init__(self, input_dim, hidden_dim, epsilon):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.learning_rate = 0.01
        self.clip_value = 0.1
        self.noise_multiplier = self.compute_noise_multiplier()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_data = tf.placeholder(tf.float32, [None, input_dim])

            # Encoder
            self.weights_encoder = {
                'encoder': tf.Variable(tf.random_normal([input_dim, hidden_dim])),
                'decoder': tf.Variable(tf.random_normal([hidden_dim, input_dim]))
            }
            self.biases_encoder = {
                'encoder': tf.Variable(tf.random_normal([hidden_dim])),
                'decoder': tf.Variable(tf.random_normal([input_dim]))
            }
            self.encoded = tf.nn.sigmoid(tf.add(tf.matmul(self.input_data, self.weights_encoder['encoder']),
                                                self.biases_encoder['encoder']))

            # Decoder
            self.decoded = tf.nn.sigmoid(tf.add(tf.matmul(self.encoded, self.weights_encoder['decoder']),
                                                 self.biases_encoder['decoder']))

            # Loss
            self.loss = tf.reduce_mean(tf.square(self.input_data - self.decoded))

            # DP-SGD Optimizer with Gradient Clipping and Gradient Noise
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
            noisy_gradients = [self.add_gradient_noise(g, self.noise_multiplier) for g in gradients]
            self.train_op = optimizer.apply_gradients(zip(noisy_gradients, variables))

            # Session
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def compute_noise_multiplier(self):
        sigma = self.clip_value / self.epsilon
        return sigma

    def add_gradient_noise(self, gradient, noise_multiplier):
        stddev = noise_multiplier * self.clip_value
        noise = tf.random_normal(tf.shape(gradient), stddev=stddev)
        return gradient + noise

    def train(self, X, epochs=100, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                self.sess.run(self.train_op, feed_dict={self.input_data: batch_X})

            if epoch % 10 == 0:
                loss = self.sess.run(self.loss, feed_dict={self.input_data: X})
                print("Epoch {}: Loss = {:.4f}".format(epoch, loss))

    def reconstruct(self, X):
        return self.sess.run(self.decoded, feed_dict={self.input_data: X})

    def get_top_anomalies(self, data, num_anomalies=None, percentage_anomalies=None):
        """
        Get the top anomalous entries from the given data.

        Parameters:
        - data (numpy.ndarray): Input data for anomaly detection.
        - num_anomalies (int): Number of top anomalies to return. Defaults to None.
        - percentage_anomalies (float): Percentage of anomalies to return (between 0 and 1). Defaults to None.

        Returns:
        - numpy.ndarray: Top anomalous entries.
        """
        # if self.autoencoder is None:
        #     raise RuntimeError("Autoencoder model not trained. Please call train() method first.")

        # Use the trained autoencoder to reconstruct the data
        reconstructed_data = self.reconstruct(data)

        # Calculate the reconstruction error for each data point
        reconstruction_errors = np.mean(np.square(data - reconstructed_data), axis=1)

        sorted_indices = np.argsort(reconstruction_errors)

        # Determine the number of anomalies to remove
        if num_anomalies is not None:
            indices_to_return = sorted_indices[-num_anomalies:]
        elif percentage_anomalies is not None:
            num_anomalies_to_return = int(len(data) * percentage_anomalies)
            indices_to_return = sorted_indices[-num_anomalies_to_return:]
        else:
            raise ValueError("Please specify either 'num_anomalies' or 'percentage_anomalies'.")

        # Get the top anomalous entries
        top_anomalies = indices_to_return

        return top_anomalies


# Example usage:
# Define parameters
# input_dim = 784  # Example: MNIST data
# hidden_dim = 64  # Example: dimension of the encoded representation
# epsilon = 1.0  # Example: privacy budget

# Initialize and train the differentially private autoencoder
# autoencoder = DP_Autoencoder(input_dim, hidden_dim, epsilon)

# Train the autoencoder using your dataset X
# autoencoder.train(X)

# Reconstruct examples using the trained autoencoder
# reconstructed_data = autoencoder.reconstruct(X)

# data = pd.read_csv('uci-epileptic.csv').to_numpy()
# data = MinMaxScaler().fit_transform(data)
#
# input_dim = data.shape[1]
# hidden_dim = int(input_dim / 2)
# epsilon = 1.0
# autoencoder = DP_Autoencoder(input_dim, hidden_dim, epsilon)
#
# autoencoder.train(data)
#
# print(autoencoder.get_top_anomalies(data, 10))