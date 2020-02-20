# Local.
from scratch.activations import Activations


class NeuralNetwork:
    """ Effective for more complex problems, such as linearly inseparable problems. e.g. XOR.

        Architecture -> Three layers.

        FULLY CONNECTED: All input nodes connected to every hidden node, every hidden node connected to every output
                        node.

                    1. Input layer with 'x' number of neurons.
                    2. Hidden layer with 'x' number of neurons.
                    3. Output layer with 'x' number of neurons.

    """

    def __init__(self, input_neurons, hidden_neurons, output_neurons, learning_rate, activation_function='relu'):
        """ Define shape of the data. """
        self.input_nodes = input_neurons
        self.hidden_nodes = hidden_neurons
        self.output_nodes = output_neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError('learning rate must be greater than 0.')
        self._learning_rate = value

    def apply_activation(self, value):
        return getattr(Activations(value), self.activation_function)()

    def feed_forward(self):
        pass
