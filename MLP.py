import random
import time

import numpy as np
from data.MNIST.preprocess import load_data

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
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a: any) -> any:
        """
        Returns the output of the network after a single pass

        :param a: input
        """
        for b, w in zip(self.biases, self.weights):
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
            z = np.dot(w, a) + b
            z_list.append(z)
            a = Sigmoid(z)
            a_list.append(a)

        # Compute the output error and gradient
        delta = (a_list[-1] - y) * dSigmoid(z_list[-1])
        dCdw[-1] = np.dot(delta, a_list[-2].transpose())
        dCdb[-1] = delta

        # Backpropagate error: for each layer l = L-1,L-2,...,2 compute error delta
        for layer in range(2, self.num_layers):
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * dSigmoid(
                z_list[-layer]
            )
            dCdw[-layer] = np.dot(delta, a_list[-layer - 1].transpose())
            dCdb[-layer] = delta

        return dCdb, dCdw

    def SGD(self, training_data, epochs, learning_rate, batch_size, test_data=None):
        m = len(training_data)
        for epoch in range(epochs):
            time1 = time.time()
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0,m,batch_size)]
            for batch in batches:
                dCdw = [np.zeros(w.shape) for w in self.weights]
                dCdb = [np.zeros(b.shape) for b in self.biases]
                for x, y in batch:
                    delta_dCdb, delta_dCdw = self.backpropagate(x, y)
                    dCdw = [nw + dnw for nw, dnw in zip(dCdw, delta_dCdw)]
                    dCdb = [nb + dnb for nb, dnb in zip(dCdb, delta_dCdb)]
                self.weights = [w - (learning_rate/len(batch)) * nw for w, nw in zip(self.weights, dCdw)]
                self.biases = [b - (learning_rate/len(batch)) * nb for b, nb in zip(self.biases, dCdb)]
            time2 = time.time()
            if test_data:
                n = len(test_data)
                print("Epoch {0}: {1} / {2}, took {3:.2f} seconds".format(
                    epoch, self.evaluate(test_data), n, time2-time1))
            else:
                print("Epoch {0} complete in {1:.2f} seconds".format(epoch, time2 - time1))

        

    def evaluate(self, test_data) -> int:
        """
        Return the number of test inputs for which the neural
        network outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def main():
    tr_d, va_d, te_d = load_data()

    net = Network([784, 30, 10])
    net.SGD(tr_d, 30, 3.0, 10, te_d)


if __name__ == "__main__":
    main()
