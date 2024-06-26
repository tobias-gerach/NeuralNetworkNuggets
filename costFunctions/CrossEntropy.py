import numpy as np


class CrossEntropy(object):
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output
        ``y``.  
        np.nan_to_num is used to ensure numerical stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer.
        Parameter ``z`` is only used for consitency with other classes.

        """
        return a - y
