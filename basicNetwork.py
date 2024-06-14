import numpy as np
import pandas as pd

from data.MNIST.preprocess import load_data

def ReLU(z: any) -> any:
    return np.maximum(0, z)

def dReLU(z: any) -> any:
    return z > 0

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
            a = ReLU(np.dot(w,a) + b)
            return a
        
    def backpropagate(self, x, y):
        """
        Return a tuple of weights dCdw and biases dCdb for each layer.
        dCdw and dCdb 

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

        # Feedforward: for each layer l = 2,3,...,L compute weighted inputs z[l] and activations a[l]
        for b, w in zip(self.biases, self.weights):
            z_list.append(np.dot(w,a) + b)
            a = ReLU(z_list[-1])
            a_list.append(a)

        # Compute the output error and gradient
        delta = (a_list[-1] - y) * dReLU(z_list[-1])
        dCdw[-1] = np.dot(delta, a_list[-2].transpose)
        dCdb[-1] = delta

        # Backpropagate error: for each layer l = L-1,L-2,...,2 compute error delta
        for layer in range(2,self.num_layers):
            delta = np.dot(self.weights[-layer+1].transpose, delta) * dReLU(z_list[-layer])
            dCdw[-layer] = np.dot(delta, a_list[-layer-1].transpose)
            dCdb[-layer] = delta

        return dCdw, dCdb



def main():
    tr_d, va_d, te_d = load_data()




if __name__ == "__main__":
    main()
