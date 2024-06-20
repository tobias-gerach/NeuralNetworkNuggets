import numpy as np


class ReLU(object):
    @staticmethod
    def fn(z):
        return np.maximum(0, z)

    @staticmethod
    def deriv(z):
        return z > 0
