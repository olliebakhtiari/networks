# Local.
from scratch.activations import Activations
from scratch.matrix import Matrix


class NeuralNetwork:
    """ Effective for more complex problems, such as linearly inseparable problems. e.g. XOR.

        Architecture -> Three layers.

        FULLY CONNECTED: All input nodes connected to every hidden node, every hidden node connected to every output
                        node.

                    1. Input layer with 'x' number of neurons.
                    2. Hidden layer with 'x' number of neurons.
                    3. Output layer with 'x' number of neurons.

    """
    IMPLEMENTED_ACTIVATIONS = ['sigmoid', 'relu', 'parametric_relu', 'leaky_relu', 'tanh']

    def __init__(
            self,
            input_neurons: int,
            hidden_neurons: int,
            output_neurons: int,
            learning_rate: float,
            activation_function='relu',
    ):
        self.input_nodes = input_neurons
        self.hidden_nodes = hidden_neurons
        self.output_nodes = output_neurons
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        # Weights between layers. Initialise randomly.
        self.input_to_hidden_weights = Matrix.construct_random_matrix(
            m=self.hidden_nodes,
            n=self.input_nodes,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )
        self.hidden_to_output_weights = Matrix.construct_random_matrix(
            m=self.output_nodes,
            n=self.hidden_nodes,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )

        # Bias values. Single column vectors. Initialise randomly.
        self.hidden_bias = Matrix.construct_random_matrix(
            m=self.hidden_nodes,
            n=1,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )
        self.output_bias = Matrix.construct_random_matrix(
            m=self.output_nodes,
            n=1,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )

    @property
    def activation_function(self):
        return self._activation_function

    @activation_function.setter
    def activation_function(self, value):
        if value not in self.IMPLEMENTED_ACTIVATIONS:
            raise NotImplementedError('cant use activation function specified.')
        self._activation_function = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError('learning rate must be greater than 0.')
        self._learning_rate = value

    def apply_activation(self, matrix: Matrix):
        for i in range(matrix.m):
            for j in range(matrix.n):
                value = matrix.data[i][j]
                matrix.data[i][j] = getattr(Activations(value), self.activation_function)()

    def feed_forward(self, inputs: list):
        input_matrix = Matrix.construct_matrix_from_lists(inputs)

        # Calculate hidden layer outputs.
        output_of_hidden_layer = (self.input_to_hidden_weights * input_matrix) + self.hidden_bias
        self.apply_activation(output_of_hidden_layer)

        # Calculate output layer outputs.
        output_of_output_layer = (self.hidden_to_output_weights * output_of_hidden_layer) + self.output_bias
        self.apply_activation(output_of_output_layer)

        return Matrix.flatten_matrix(output_of_output_layer)


if __name__ == '__main__':
    nn = NeuralNetwork(2, 2, 1, learning_rate=0.1)
    print(nn.feed_forward([[1], [0]]))

