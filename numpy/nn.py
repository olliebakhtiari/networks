import numpy as np
from scipy.stats import truncnorm


class NeuralNetwork:
    def __init__(self, input_layer_nodes: int, hidden_layer_nodes: int, output_layer_nodes: int, learning_rate: float):
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes
        self.learning_rate = learning_rate

        self._initialise_weight_matrices()

    @classmethod
    def _truncated_normal(cls, mean=0, sd=1, low=0., upp=10):
        """ 'truncated_normal' is ideal for weight initialisation. It is a good idea to choose random values from within
            the interval
                                                        (−1/sqrt(n), 1/sqrt(n))
        """
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def _initialise_weight_matrices(self):
        rad = 1 / np.sqrt(self.input_layer_nodes)
        x = self._truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_input_to_hidden = x.rvs((self.hidden_layer_nodes, self.input_layer_nodes))

        rad = 1 / np.sqrt(self.hidden_layer_nodes)
        x = self._truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_to_output = x.rvs((self.output_layer_nodes, self.hidden_layer_nodes))


if __name__ == '__main__':
    nn = NeuralNetwork(3, 4, 2, 0.1)
    print(nn.weights_input_to_hidden)
    print(nn.weights_hidden_to_output)
