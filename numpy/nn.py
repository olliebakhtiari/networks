# Third-party.
import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit, logit


def get_data():
    data1 = [
        ((3, 4), (0.99, 0.01)), ((4.2, 5.3), (0.99, 0.01)),
        ((4, 3), (0.99, 0.01)), ((6, 5), (0.99, 0.01)),
        ((4, 6), (0.99, 0.01)), ((3.7, 5.8), (0.99, 0.01)),
        ((3.2, 4.6), (0.99, 0.01)), ((5.2, 5.9), (0.99, 0.01)),
        ((5, 4), (0.99, 0.01)), ((7, 4), (0.99, 0.01)),
        ((3, 7), (0.99, 0.01)), ((4.3, 4.3), (0.99, 0.01)),
    ]
    data2 = [
        ((-3, -4), (0.01, 0.99)), ((-2, -3.5), (0.01, 0.99)),
        ((-1, -6), (0.01, 0.99)), ((-3, -4.3), (0.01, 0.99)),
        ((-4, -5.6), (0.01, 0.99)), ((-3.2, -4.8), (0.01, 0.99)),
        ((-2.3, -4.3), (0.01, 0.99)), ((-2.7, -2.6), (0.01, 0.99)),
        ((-1.5, -3.6), (0.01, 0.99)), ((-3.6, -5.6), (0.01, 0.99)),
        ((-4.5, -4.6), (0.01, 0.99)), ((-3.7, -5.8), (0.01, 0.99)),
    ]
    data_set = data1 + data2
    np.random.shuffle(data_set)

    return data_set


def get_labelled_data():
    class_1 = [
        (3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
        (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3),
    ]
    class_2 = [
        (-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
        (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
        (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8),
    ]
    labelled_data = []
    for a in class_1:
        labelled_data.append([a, [1, 0]])
    for a in class_2:
        labelled_data.append([a, [0, 1]])
    np.random.shuffle(labelled_data)

    return labelled_data


@np.vectorize
def sigmoid(x):
    """ https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.vectorize.html """
    return 1 / (1 + np.e ** -x)


########################################################################################################################


class NeuralNetwork:
    def __init__(
            self,
            input_layer_nodes: int,
            hidden_layer_nodes: int,
            output_layer_nodes: int,
            learning_rate: float,
            bias=None,
    ):
        self.input_layer_nodes = input_layer_nodes
        self.hidden_layer_nodes = hidden_layer_nodes
        self.output_layer_nodes = output_layer_nodes
        self.learning_rate = learning_rate
        self.bias = bias
        self._initialise_weight_matrices()

    @classmethod
    def _truncated_normal(cls, mean=0, sd=1, low=0., upp=10):
        """ 'truncated_normal' is ideal for weight initialisation. It is a good idea to choose random values from within
            the interval.
                                                        (−1/sqrt(n), 1/sqrt(n))
        """
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def _initialise_weight_matrices(self):
        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.input_layer_nodes + bias_node)
        x = self._truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_input_to_hidden = x.rvs((self.hidden_layer_nodes, self.input_layer_nodes + bias_node))

        rad = 1 / np.sqrt(self.hidden_layer_nodes + bias_node)
        x = self._truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_to_output = x.rvs((self.output_layer_nodes, self.hidden_layer_nodes + bias_node))

    def feedforward(self, input_vector):
        """ The expit function, also known as the logistic sigmoid function, is defined as expit(x) = 1/(1+exp(-x)).
            It is the inverse of the logit function.
        """

        # input_vector can be tuple, list or ndarray
        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_input_to_hidden, input_vector)
        output_vector = expit(output_vector)
        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))
        output_vector = np.dot(self.weights_hidden_to_output, output_vector)
        output_vector = expit(output_vector)

        return output_vector

    def train(self, input_vector, target_vector):
        if self.bias:
            # adding bias node to the end of the input_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector_1 = np.dot(self.weights_input_to_hidden, input_vector)
        output_vector_hidden = expit(output_vector_1)

        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector_2 = np.dot(self.weights_hidden_to_output, output_vector_hidden)
        output_vector_network = expit(output_vector_2)

        output_errors = target_vector - output_vector_network

        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_to_output += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)

        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_input_to_hidden += self.learning_rate * x


if __name__ == '__main__':
    nn = NeuralNetwork(
        input_layer_nodes=2,
        hidden_layer_nodes=10,
        output_layer_nodes=2,
        learning_rate=0.1,
        bias=True,
    )

    # 1st env.
    data = get_data()
    size_of_learn_sample = int(len(data) * 0.9)
    learn_data = data[:size_of_learn_sample]
    test_data = data[-size_of_learn_sample:]
    for i in range(size_of_learn_sample):
        point, label = learn_data[i][0], learn_data[i][1]
        nn.train(point, label)
    for j in range(size_of_learn_sample):
        point, label = learn_data[j][0], learn_data[j][1]
        cls1, cls2 = nn.feedforward(point)
        print(point, cls1, cls2, end=": ")
        if cls1 > cls2:
            if label == (0.99, 0.01):
                print("class_1 correct", label)
            else:
                print("class_2 incorrect", label)
        else:
            if label == (0.01, 0.99):
                print("class_1 correct", label)
            else:
                print("class_2 incorrect", label)

    # 2nd env.
    labelled = get_labelled_data()
    print(labelled[:10])
    data, labels = zip(*labelled)
    labels = np.array(labels)
    data = np.array(data)
    for _ in range(20):
        for i in range(len(data)):
            nn.train(data[i], labels[i])
    for i in range(len(data)):
        print(labels[i])
        print(nn.feedforward(data[i]))
