import gzip
import pickle

import numpy as np


def load_data() -> tuple[any, any, any]:
    """
    Returns the MNIST data containing the training data, the validation data, and the test data.
    The training data consists of 50,000 entries, while the validation and test data consists of 10,000 entries each.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    """

    f = gzip.open("./data/MNIST/mnist.pkl.gz", "rb")
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def one_encode_vector(j: int):
    """
    Returns a 10-dimensional vector with 1.0 in jth position and 0.0 in the others.
    """

    v =np.zeros((10, 1))
    v[j] = 1.0
    return v
