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

    IMPLEMENTED_ACTIVATIONS = [
        'sigmoid',
        'reverse_sigmoid',
        'relu',
        'parametric_relu',
        'leaky_relu',
        'tanh',
    ]

    def __init__(
            self,
            input_layer_nodes: int,
            hidden_layer_nodes: int,
            output_layer_nodes: int,
            learning_rate: float,
            activation_function='sigmoid',
            reverse_activation='reverse_sigmoid',
    ):
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.reverse_activation = reverse_activation

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

    def apply_or_reverse_activation(self, matrix: Matrix, reverse=False, return_new_matrix=False):
        func = self.activation_function
        if reverse:
            func = self.reverse_activation
        if return_new_matrix:
            new_matrix = Matrix(matrix.m, matrix.n, zeroes=True)
            for i in range(new_matrix.m):
                for j in range(new_matrix.n):
                    value = matrix.data[i][j]
                    new_matrix.data[i][j] = getattr(Activations(value), func)()
            return new_matrix

        # Modify given matrix if return_new_matrix is False.
        for i in range(matrix.m):
            for j in range(matrix.n):
                value = matrix.data[i][j]
                matrix.data[i][j] = getattr(Activations(value), func)()

    def approximate(self, inputs: list):
        input_matrix = Matrix.construct_matrix_from_lists(inputs)

        # Calculate layer outputs. Outputs(layer) = ReLU(Weights(layer) * Inputs(layer) + Biases(layer)).
        output_of_hidden_layer = (self.input_to_hidden_weights * input_matrix) + self.hidden_bias
        self.apply_or_reverse_activation(output_of_hidden_layer)

        output_of_output_layer = (self.hidden_to_output_weights * output_of_hidden_layer) + self.output_bias
        self.apply_or_reverse_activation(output_of_output_layer)

        return output_of_output_layer

    def learn(self, inputs, targets):
        """ Using Sigmoid as activation:
                                                                 --derivative--
            DELTA HO Weights[i][j] = learning rate * error * (output * (1 - output)) * hidden(transposed)
            DELTA IH Weights[i][j] = learning rate * hidden error * (hidden * (1 - hidden)) * input(transposed)

        """
        input_matrix = Matrix.construct_matrix_from_lists(inputs)

        # Calculate layer outputs. Outputs(layer) = ReLU(Weights(layer) * Inputs(layer) + Biases(layer)).
        output_of_hidden_layer = (self.input_to_hidden_weights * input_matrix) + self.hidden_bias
        self.apply_or_reverse_activation(output_of_hidden_layer)

        output_of_output_layer = (self.hidden_to_output_weights * output_of_hidden_layer) + self.output_bias
        self.apply_or_reverse_activation(output_of_output_layer)

        targets_matrix = Matrix.construct_matrix_from_lists(targets)

        # Output layer errors.
        output_errors = targets_matrix - output_of_output_layer

        # Output layer gradients.
        oe_gradients = self.apply_or_reverse_activation(
            matrix=output_of_output_layer,
            reverse=True,
            return_new_matrix=True,
        )
        grad_o = oe_gradients * output_errors
        grad_o.scalar_mult(self.learning_rate)

        # Get deltas.
        hidden_outputs_transposed = output_of_hidden_layer.get_transpose()
        ho_weights_deltas = grad_o * hidden_outputs_transposed

        # Adjust weights and bias.
        self.hidden_to_output_weights = self.hidden_to_output_weights + ho_weights_deltas
        self.output_bias = self.output_bias + grad_o

        # Hidden layer errors.
        ho_weights_transposed = self.hidden_to_output_weights.get_transpose()
        hidden_errors = ho_weights_transposed * output_errors

        # Hidden layer gradients.
        he_gradients = self.apply_or_reverse_activation(
            matrix=output_of_hidden_layer,
            reverse=True,
            return_new_matrix=True,
        )
        grad_h = he_gradients * hidden_errors
        grad_h.scalar_mult(self.learning_rate)

        # Get input to hidden deltas.
        inputs_transposed = input_matrix.get_transpose()
        ih_weights_deltas = grad_h * inputs_transposed

        # Adjust weights and bias.
        self.input_to_hidden_weights = self.input_to_hidden_weights + ih_weights_deltas
        self.hidden_bias = self.hidden_bias + grad_h

        return self


if __name__ == '__main__':
    i_ = [[1], [0]]
    t_ = [[1]]
    nn = NeuralNetwork(2, 2, 1, learning_rate=0.1)
    # print(Matrix.flatten_matrix(nn.approximate(i_)))
    print(nn.learn(i_, t_))

