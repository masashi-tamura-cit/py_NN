import random
from consts import *
import numpy as np
import math
# from main import view_data as vd
import sys


class Weights:
    def __init__(self, weight: np.ndarray, lead_layer, lag_layer):
        self.weight = weight  # 重み
        self.lag_layer = lag_layer  # Layer class object
        self.lead_layer = lead_layer  # Layer class object
        self.gradients = np.array([])  # 本当は2次元で重みと同じ構造

    def calc_grad(self):
        self.gradients = np.dot(self.lag_layer.delta.T, self.lead_layer.activated_values).T / BATCH_SIZE

    def sgd(self):
        self.weight = self.weight - (ETA * self.gradients)


class Layer:
    def __init__(self, amount: int):
        self.node_amount = amount
        self.net_values = np.array([])
        self.activated_values = np.array([])
        self.activated_values = np.array([])
        self.lead_weights = None
        self.lag_weights = None
        self.delta = np.array([])
        self.test_nodes = np.array([])
        self.sd = []

    def normalize(self):
        var = []
        ave = []
        for v in self.net_values.T:
            var.append(np.var(v) + 0.0001)  # avoid 0 divide
            ave.append(np.average(v))
        self.sd = np.sqrt(var)
        self.net_values = (self.net_values - ave) / self.sd

    def activate(self):
        batch_size = self.net_values.shape[0]
        # self.activated_values = 1/(1 + np.exp(-self.net_values))
        self.activated_values = np.where(self.net_values < 0, 0, self.net_values)
        self.activated_values = np.reshape(np.append(self.activated_values.T, np.ones(batch_size)),
                                           (self.node_amount + 1, batch_size)).T

    def calc_net_values(self):
        self.lag_weights.lag_layer.net_values = np.dot(self.activated_values, self.lag_weights.weight)

    def calc_delta(self, labels: list):
        norm_diff = (1/(BATCH_SIZE * self.sd))*(BATCH_SIZE - pow(self.net_values, 2) - 1)
        round_z = self.lag_weights.lag_layer.delta.dot(self.lag_weights.weight[:-1].T) * norm_diff
        self.delta = round_z * np.where(self.net_values <= 0, 0, 1)


class SigmoidLayer(Layer):
    def normalize(self):
        pass

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = 1/(1 + np.exp(-self.net_values))
        self.activated_values = np.reshape(np.append(self.activated_values.T, np.ones(batch_size)),
                                           (self.node_amount + 1, batch_size)).T

    def calc_delta(self, labels: list):
        round_z = np.dot(self.lag_weights.lag_layer.delta, self.lag_weights.weight[:-1].T)
        self.delta = round_z * (1 - self.activated_values.T[:-1].T) * self.activated_values.T[:-1].T


class SoftMaxLayer(Layer):
    def normalize(self):
        pass  # ReLUのときはコメントアウト

    def activate(self):
        value_arr = []
        for values in self.net_values:
            e = np.exp(values - values.max())
            value_arr.append(e / np.sum(e))
        self.activated_values = np.array(value_arr)

    def calc_delta(self, labels: list) -> np.array:
        # norm_diff = (1/(BATCH_SIZE * self.sd))*(BATCH_SIZE - pow(self.net_values, 2) - 1)
        norm_diff = 1
        delta_arr = np.array(list(self.activated_values))  # avoid_update_activated_value
        for i, delta in zip(labels, delta_arr):
            delta[i] = delta[i] - 1
        self.delta = np.array(delta_arr) * norm_diff

    def calc_net_values(self):
        pass


class NetWork:
    def __init__(self, hidden_layer: int, input_dim: int):
        self.hidden_layer = hidden_layer
        if hidden_layer < 2:
            self.hidden_layer = 2

        self.layers = []
        if ACTIVATE == SIGMOID:
            self.layers.append(SigmoidLayer(input_dim))
            for i in range(hidden_layer - 1):
                self.layers.append(SigmoidLayer(MD1))
            self.layers.append(SigmoidLayer(MD2))
        if ACTIVATE == ReLU:
            self.layers.append(Layer(input_dim))
            for i in range(hidden_layer - 1):
                self.layers.append(Layer(MD1))
            self.layers.append(Layer(MD2))

        self.layers.append(SoftMaxLayer(CLASS_NUM))
        self.weights = []
        self.__init_weight()
        self.data_error = 0.0
        self.data = np.array([])
        self.labels = np.array([])

    def __init_weight(self):
        for lead, lag in zip(self.layers[:-1], self.layers[1:]):
            arr = np.random.randn(lead.node_amount + 1, lag.node_amount) / np.sqrt(lead.node_amount + 1)
            weight = Weights(arr, lead, lag)
            lead.lag_weights = weight
            lag.lead_weights = weight
            self.weights.append(weight)
        self.layers[-1].lead_weights.weight = self.layers[-1].lead_weights.weight * np.sqrt(2.0)

    def training(self, train_data, labels):
        self.data = train_data
        self.labels = labels
        data_size = np.array(self.labels).size
        self.data_error = 0
        for item, key in zip(self.data, self.labels):
            # input_data_to_input_layer
            self.layers[0].net_values = np.array(item)
            # forward operation
            for layer in self.layers:
                layer.normalize()
                layer.activate()
                layer.calc_net_values()
            # calc_error
            for v, label in zip(self.layers[-1].activated_values, key):
                self.data_error += -math.log(v[label], math.e) / data_size
            # back propagation
            for layer in reversed(self.layers):
                layer.calc_delta(key)
            for weight in self.weights:
                weight.calc_grad()
                weight.sgd()  # optimizer

    def test(self, test_data: np.array, test_labels: list) -> tuple:
        accuracy = 0.0
        error_average = 0.0
        self.data = test_data
        self.layers[0].net_values = test_data
        self.labels = test_labels

        data_amount = len(test_labels)
        for layer in self.layers:
            layer.normalize()
            layer.activate()
            layer.calc_net_values()
        for v, label in zip(self.layers[-1].activated_values, self.labels):
            if np.argmax(v) == label:
                accuracy += 1.0/data_amount
            error_average += -math.log(v[label], math.e) / data_amount
        return accuracy, error_average
