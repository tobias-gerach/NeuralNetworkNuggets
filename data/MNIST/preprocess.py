import gzip
import pickle

import numpy as np


def load_data():
    """
    Returns the MNIST data containing the training data, the validation data, and the test data.
    The training data consists of 50,000 entries, while the validation and test data consists of 10,000 entries each.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    In case of the validation and test data, ``x`` is a 784-dimensional array containing the input data
    and ``y`` is the corresponding classification.
    """

    f = gzip.open("./data/MNIST/mnist.pkl.gz", "rb")
    u = pickle._Unpickler(f)
    u.encoding = "latin1"
    training_data, validation_data, test_data = u.load()
    f.close()
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [one_encode_vector(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_data = list(zip(validation_inputs, validation_data[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))
    return (training_data, validation_data, test_data)

def one_encode_vector(j: int):
    """
    Returns a 10-dimensional vector with 1.0 in jth position and 0.0 in the others.
    """

    v =np.zeros((10, 1))
    v[j] = 1.0
    return v
