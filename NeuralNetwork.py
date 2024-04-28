import random

import numpy as np


class Relu:
    def __init__(self):
        self.output = None
        self.div = None
        self.input = None

    def forward(self, layer_output):
        self.input = layer_output
        for i in range(len(layer_output)):
            layer_output[i] = max(0, layer_output[i])
        self.output = layer_output

    def backward(self, layer_output):
        relu_output = layer_output.copy()
        for i in range(len(layer_output)):
            if layer_output[i] <= 0:
                relu_output[i] = 0
            else:
                relu_output[i] = 1

        self.div = relu_output


class Softmax:
    def __init__(self):
        self.output = None
        self.div = None
        self.input = None

    def forward(self, layer_output):
        self.input = layer_output
        exp_values = np.exp(layer_output - np.max(layer_output))
        probabilities = exp_values / np.sum(exp_values)

        self.output = probabilities

    def backward(self, _):
        self.div = 1


class Sigmoid:
    def __init__(self):
        self.output = None
        self.div = None
        self.input = None

    def forward(self, layer_output):
        self.input = layer_output

    def backward(self, layer_output):
        pass


class Tanh:
    def __init__(self):
        self.output = None
        self.div = None
        self.input = None

    def forward(self, layer_output):
        self.input = layer_output

    def backward(self, layer_output):
        pass


class Layer:
    def __init__(self, weights, activation_function=None):
        self.weights = weights
        self.dropout_table = np.ones(self.weights.shape[1])
        self.dropout_multiplayer = 1
        self.dropout = False
        self.activation_function = activation_function
        self.input = None
        self.output = None

    def forward(self, input: list):
        self.input = input

        if self.dropout:
            self.output = np.dot(self.weights*self.dropout_table, input) * self.dropout_multiplayer
        else:
            self.output = np.dot(self.weights, input)

        if self.activation_function is None:
            return

        self.activation_function.forward(self.output)
        self.activation_function.backward(self.output)

        self.output = self.activation_function.output

    def backward(self, delta, alpha):
        weight_delta = np.outer(delta, self.input)

        if self.dropout:
            weight_delta = weight_delta*self.dropout_table

        self.adjust_weights(weight_delta, alpha)

    def dropout_on(self):
        self.dropout = True

        n = self.weights.shape[1]
        print(n)
        for _ in range(n//2):
            self.dropout_table[random.randint(0, n-1)] = 0

        self.dropout_multiplayer = n/sum(self.dropout_table)

    def dropout_off(self):
        self.dropout = False
        self.dropout_table = np.ones(self.weights.shape[1])

    def adjust_weights(self, weight_delta, alpha):
        self.weights = self.weights - (alpha * weight_delta)

    def get_input_size(self):
        return len(self.weights)


class NeuralNetwork:
    def __init__(self, alpha, input_size, dropout_probability=0.3):
        self.alpha = alpha
        self.layers = []
        self.series = []
        self.goals = []
        self.input_size = input_size
        self.dropout_probability = dropout_probability

    def add_layer(self, n, weight_range=(-0.1, 0.1), activation_function=Relu()):
        if not len(self.layers) == 0:
            last_input_size = self.layers[len(self.layers)-1].get_input_size()
        else:
            last_input_size = self.input_size
        new_weights = np.random.uniform(weight_range[0], weight_range[1], size=(n, last_input_size))

        new_layer = Layer(new_weights, activation_function)

        self.layers.append(new_layer)

    def random_dropout_on(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                return

            if random.random() < self.dropout_probability:
                layer.dropout_on()

    def random_dropout_off(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                return

            layer.dropout_off()

    def fit(self, _input: list, expected_output: list, random_dropout=False):
        if random_dropout:
            self.random_dropout_on()

        layer_input = _input

        for layer in self.layers:
            layer.forward(layer_input)
            layer_input = layer.output

        output_delta = (2 / len(layer_input)) * (np.subtract(layer_input, expected_output))

        delta = output_delta
        for layer in reversed(self.layers):
            delta = delta * layer.activation_function.div
            layer.backward(delta, self.alpha)

            delta = np.dot(np.transpose(layer.weights), delta)

        if random_dropout:
            self.random_dropout_off()

    def predict(self, input: list):
        layer_input = input
        for layer in self.layers:
            layer.forward(layer_input)
            layer_input = layer.output

        return layer_input

    def save_weights(self, file_name):
        pass

    def load_weights(self, file_name):
        pass

    def do_epochs(self, epochs):
        for epoch_num in range(0, epochs):
            correct = 0
            for series_num in range(0, len(self.series)):
                output_ = self.fit(self.series[series_num], self.goals[series_num])
                if np.argmax(output_) == np.argmax(self.goals[series_num]):
                    correct += 1
            print("Epoch:", epoch_num + 1, "Accuracy:", correct / len(self.series) * 100)
        return self.weights_list