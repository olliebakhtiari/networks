# Python standard.
import random
from typing import List

# Third-party.
import numpy as np

# Local.
from scratch.activations import Activations


class Perceptron:
    """ Single layer Neural Network. Linear classifier (binary). For Linearly separable problems."""

    def __init__(self, inputs: List[float], activation_fn_name: str, learning_rate: float):
        # Random initialization for weights so different hidden units learn different things.
        self.bias = 1
        self.inputs = inputs + [self.bias]
        self.weights = np.array([random.random() for _ in range(len(inputs) + self.bias)])
        self.activation_fn_name = activation_fn_name
        self.learning_rate = learning_rate

    @staticmethod
    def sign(x: float):
        return 1 if x >= 0 else 0

    def feed_forward(self):
        """ Z(l) = W(l)X(l) + B(l). Linear forward function. Bias is added on creation of Perceptron object. """
        sum_ = 0
        for i in range(len(self.weights)):
            sum_ += self.inputs[i] * self.weights[i]

        return getattr(Activations(sum_), self.activation_fn_name)()

    def feed_forward_and_adjust_weights(self, target: float):
        """ delta weight = error * input * learning rate. """
        estimate = self.feed_forward()
        error = target - estimate
        for i in range(len(self.weights)):
            self.weights[i] += error * self.inputs[i] * self.learning_rate


if __name__ == '__main__':
    perceptron = Perceptron(
        inputs=[-1, 0.5, 0.25, -0.75, 0.99, 0.01],
        activation_fn_name='sigmoid',
        learning_rate=0.01,
    )
    print(perceptron.feed_forward())
