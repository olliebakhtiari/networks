# Third-party.
import numpy as np


class Activations:
    def __init__(self, x: float):
        self.x = x  # input.

    def sigmoid(self):
        """ Mathematical function having a characteristic "S"-shaped curve or sigmoid curve. For Feedforward/Forward
            propagation.

            Range: (0,1).

        :return: f(x).
        """
        return 1 / (1 + np.exp(-self.x))

    def reverse_sigmoid(self):
        """ Derivative of Sigmoid.

        :return: dsigmoid.
        """
        s = self.sigmoid()
        ds = s * (1 - s)

        return self.x * ds

    def relu(self):
        """ Linear (identity) for all positive values, and zero for all negative values.

            Range: [0, infinity).

        :return: f(x).
        """
        return np.max(self.x, 0)

    def leaky_relu(self):
        """ Allow a small, positive gradient when the unit is not active.

            Range: (-infinity, infinity).

        :return: f(x).
        """
        return self.x if self.x > 0 else 0.01 * self.x

    def parametric_relu(self, a: float):
        """ Type of leaky ReLU that, instead of having a predetermined slope like 0.01, makes it a parameter.

            Range: (-infinity, infinity).

        :param a: coefficient of leakage, typically 0.01 <= a <= 0.99
        :return: f(x).
        """
        return self.x if self.x > 0 else a * self.x

    def elu(self):
        """ Has a small slope for negative values. Instead of a straight line, it uses a log curve.

            Range: (alpha, infinity).

        :return:
        """
        raise NotImplementedError

    def tanh(self):
        """ Tanh function is also sigmoidal (“s”-shaped), but instead outputs values that range (-1, 1). Thus strongly
            negative inputs to the tanh will map to negative outputs. Additionally, only zero-valued inputs are mapped to
            near-zero outputs.

            Range: (-1,1).

        :return: f(x).
        """
        return np.exp(self.x) - np.exp(-self.x) / (np.exp(self.x) + np.exp(-self.x))
