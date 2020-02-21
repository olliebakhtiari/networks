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
            input_layer_nodes: int,
            hidden_layer_nodes: int,
            output_layer_nodes: int,
            learning_rate: float,
            activation_function='relu',
    ):
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        # Weights between layers. Initialise randomly.
        self.input_to_hidden_weights = Matrix.construct_random_matrix(
            m=self.hidden_layer_nodes,
            n=self.input_layer_nodes,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )
        self.hidden_to_output_weights = Matrix.construct_random_matrix(
            m=self.output_layer_nodes,
            n=self.hidden_layer_nodes,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )

        # Bias values. Single column vectors. Initialise randomly.
        self.hidden_bias = Matrix.construct_random_matrix(
            m=self.hidden_layer_nodes,
            n=1,
            low=-1,
            high=1,
            d_type='float',
            precision=3,
        )
        self.output_bias = Matrix.construct_random_matrix(
            m=self.output_layer_nodes,
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

    def feedforward(self, inputs: list):
        input_matrix = Matrix.construct_matrix_from_lists(inputs)

        # Calculate layer outputs. Outputs(layer) = Weights(layer) * Inputs(layer) + Biases(layer).
        output_of_hidden_layer = (self.input_to_hidden_weights * input_matrix) + self.hidden_bias
        self.apply_activation(output_of_hidden_layer)

        output_of_output_layer = (self.hidden_to_output_weights * output_of_hidden_layer) + self.output_bias
        self.apply_activation(output_of_output_layer)

        return output_of_output_layer

    def backpropagation(self, inputs, targets):
        outputs = self.feedforward(inputs)
        targets_matrix = Matrix.construct_matrix_from_lists(targets)

        # Output layer errors.
        output_errors = targets_matrix - outputs

        # Hidden layer errors.
        ho_weights_transposed = self.hidden_to_output_weights.get_transpose()
        hidden_errors = ho_weights_transposed * output_errors

        return hidden_errors, output_errors


if __name__ == '__main__':
    i = [[1], [0]]
    t = [[1]]
    nn = NeuralNetwork(2, 2, 1, learning_rate=0.1)
    # print(Matrix.flatten_matrix(nn.feedforward(i)))
    # print(nn.learn(i, t))

