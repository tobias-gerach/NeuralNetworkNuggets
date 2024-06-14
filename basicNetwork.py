import random
import time

import numpy as np

from data.MNIST.preprocess import load_data


def ReLU(z: any) -> any:
    return np.maximum(0, z)


def dReLU(z: any) -> any:
    return z > 0


def Sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def dSigmoid(z):
    """Derivative of the sigmoid function."""
    return Sigmoid(z) * (1 - Sigmoid(z))


class Network(object):
    """
    Multi-layer perceptron
    """

    def __init__(self, sizes: list) -> None:
        """
        Initializes the network with random weights and biases.
        Input layer does not have weights and biases.

        :param sizes: list containing the number of neurons per layer, e.g. [784 16 10]
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a: any) -> any:
        """
        Returns the output of the network after a single pass

        :param a: input
        """
        for b, w in zip(self.biases, self.weights):
            # a = ReLU(np.dot(w, a) + b)
            a = Sigmoid(np.dot(w, a) + b)
            return a

    def backpropagate(self, x, y):
        """
        Return the cost gradient w.r.t. weights dCdw and biases dCdb

        :param x: input data/features
        :param y: labels
        """
        # Initialize lists
        dCdw = [np.zeros(w.shape) for w in self.weights]
        dCdb = [np.zeros(b.shape) for b in self.biases]
        a_list = []
        z_list = []

        # Set activation for input layer as x
        a = x
        a_list.append(a)

        # Feedforward: for each layer l = 2,3,...,L compute weighted inputs z[l] and activations a[l]
        for b, w in zip(self.biases, self.weights):
            z_list.append(np.dot(w, a) + b)
            # a = ReLU(z_list[-1])
            a = Sigmoid(z_list[-1])
            a_list.append(a)

        # Compute the output error and gradient
        # delta = (a_list[-1] - y) * dReLU(z_list[-1])
        delta = (a_list[-1] - y) * dSigmoid(z_list[-1])
        dCdw[-1] = np.dot(delta, a_list[-2].transpose())
        dCdb[-1] = delta

        # Backpropagate error: for each layer l = L-1,L-2,...,2 compute error delta
        for layer in range(2, self.num_layers):
            # delta = np.dot(self.weights[-layer + 1].transpose(), delta) * dReLU(z_list[-layer])
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * dSigmoid(
                z_list[-layer]
            )
            dCdw[-layer] = np.dot(delta, a_list[-layer - 1].transpose())
            dCdb[-layer] = delta

        return dCdw, dCdb

    def gradientDescent(self, training_data, epochs, learning_rate, test_data=None):
        for epoch in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            dCdw = [np.zeros(w.shape) for w in self.weights]
            dCdb = [np.zeros(b.shape) for b in self.biases]
            for x, y in training_data:
                delta_dCdw, delta_dCdb = self.backpropagate(x, y)
                dCdw = [nw + dnw for nw, dnw in zip(dCdw, delta_dCdw)]
                dCdb = [nb + dnb for nb, dnb in zip(dCdb, delta_dCdb)]
            self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, dCdw)]
            self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, dCdb)]
            time2 = time.time()
            print("Epoch {0} complete in {1:.2f} seconds".format(epoch, time2 - time1))

        if test_data:
            n = len(test_data)
            print(
                "Accuracy after {0} epochs: {1} / {2}".format(
                    epoch, self.evaluate(test_data), n
                )
            )

    def evaluate(self, test_data) -> int:
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def main():
    tr_d, va_d, te_d = load_data()

    net = Network([784, 16, 10])
    net.gradientDescent(tr_d, 10, 3.0, te_d)


if __name__ == "__main__":
    main()
