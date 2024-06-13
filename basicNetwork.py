import numpy as np
import pandas as pd

from data.MNIST.preprocess import load_data

def ReLU(z: any) -> any:
    return np.maximum(0, z)

def dReLU(z: any) -> any:
    return z > 0

class Network(object):
    """
    Simple network with a single hidden layer
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
            a = ReLU(np.dot(w,a) + b)
            return a


def main():
    tr_d, va_d, te_d = load_data()

    sizes = [784,16,10]
    w = [np.random.randn(y, x) for y, x in zip(sizes[1:], sizes[:-1])]
    for y, x in zip(sizes[1:], sizes[:-1]):
        print(y,x)



if __name__ == "__main__":
    main()
