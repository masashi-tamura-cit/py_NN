import random
from consts import *
import numpy as np
import math
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
        # self.sd = np.array([])
        self.sd = 0
        self.test_nodes = np.array([])

    def normalize(self):
        # self.sd = []
        # ave = []
        # for v in self.net_values.T:
        #    self.sd.append(np.sqrt(np.var(v)))
        #    ave.append(np.ave(v))
        # self.net_values = (self.net_values.T - ave) / self.sd
        self.sd = np.sqrt(np.var(self.net_values))
        self.net_values = (self.net_values - np.average(self.net_values)) / self.sd

    def activate(self):
        self.activated_values = np.where(self.net_values < 0, 0, self.net_values)
        self.activated_values = np.reshape(np.append(self.activated_values.T, np.ones(BATCH_SIZE)),
                                           (self.node_amount + 1, BATCH_SIZE)).T

    def calc_net_values(self):
        self.lag_weights.lag_layer.net_values = np.dot(self.activated_values, self.lag_weights.weight)

    def calc_delta(self, labels: list):
        nsd = self.net_values.size * self.sd
        norm_diff = (self.net_values * self.net_values) * (-1 / nsd) + ((1 - self.net_values.size)/nsd)
        round_z = np.dot(self.lag_weights.lag_layer.delta, self.lag_weights.weight[:-1].T)
        self.delta = round_z * np.where(round_z < 0, 0, 1) * norm_diff

    def forward(self):
        v = self.test_nodes
        v = (v - np.average(v)) / np.sqrt(np.var(v))
        v = np.where(v < 0, 0, v)  # activate
        v = np.append(v, 1)  # add_bias_node
        self.lag_weights.lag_layer.test_nodes = np.dot(v, self.lag_weights.weight)  # calc_lag_values


class SoftMaxLayer(Layer):
    def activate(self):
        value_arr = []
        for values in self.net_values:
            e = np.exp(values - values.max())
            value_arr.append(e / np.sum(e))
        self.activated_values = np.array(value_arr)

    def calc_delta(self, labels: list):
        delta_arr = self.activated_values
        nsd = self.net_values.size * self.sd
        norm_diff = ((self.net_values * self.net_values) * (-1 / nsd)) + ((1 - self.net_values.size)/nsd)
        for i, delta in zip(labels, delta_arr):
            delta[i] = delta[i] - 1
        self.delta = np.array(delta_arr) * norm_diff

    def calc_net_values(self):
        pass

    def forward(self):
        v = self.test_nodes
        v = (v - np.average(v)) / np.sqrt(np.var(v))
        v = np.exp(v - v.max())
        self.test_nodes = v / np.sum(v)


class NetWork:
    def __init__(self, hidden_layer: int):
        self.hidden_layer = hidden_layer
        if hidden_layer < 2:
            self.hidden_layer = 2

        self.layers = []
        self.layers.append(Layer(DIM))
        for i in range(hidden_layer - 1):
            self.layers.append(Layer(MD1))
        self.layers.append(Layer(MD2))
        self.layers.append(SoftMaxLayer(CLASS_NUM))
        self.weights = []
        self.__init_weight()
        self.train_data_error = 0.0
        self.train_data = np.array([])
        self.train_labels = np.array([])

    def __init_weight(self):
        for lead, lag in zip(self.layers[:-1], self.layers[1:]):
            arr = np.random.randn(lead.node_amount + 1, lag.node_amount) / np.sqrt(lead.node_amount + 1)
            weight = Weights(arr, lead, lag)
            lead.lag_weights = weight
            lag.lead_weights = weight
            self.weights.append(weight)
        self.layers[-1].lead_weights.weight = self.layers[-1].lead_weights.weight * np.sqrt(2.0)

    def training(self):
        data_size = np.array(self.train_labels).size
        self.train_data_error = 0
        for item, key in zip(self.train_data, self.train_labels):
            # input_data_to_input_layer
            self.layers[0].net_values = np.array(item)
            # forward operation
            for layer in self.layers:
                layer.normalize()
                layer.activate()
                layer.calc_net_values()
            # calc_error
            for v, data_label in zip(self.layers[-1].activated_values, key):
                self.train_data_error += -math.log(v[data_label], math.e) / data_size
            # back propagation
            for layer in reversed(self.layers):
                layer.calc_delta(key)
            for weight in self.weights:
                weight.calc_grad()
                weight.sgd()  # optimizer

    def test(self, test_data: np.array, test_labels: list) -> tuple:
        recognition_late = 0
        error_average = 0
        for t, data in zip(test_labels, test_data):
            self.layers[0].test_nodes = np.array(data)
            for layer in self.layers:
                layer.forward()
            if self.layers[-1].test_nodes.argmax() == t:
                recognition_late += 1
            error_average += -(math.log(self.layers[-1].test_nodes[t], math.e))
        return recognition_late / len(test_labels), error_average / len(test_labels)
