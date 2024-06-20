import numpy as np


class Sigmoid(object):
    @staticmethod
    def fn(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def deriv(z):
        return Sigmoid.fn(z) * (1 - Sigmoid.fn(z))
