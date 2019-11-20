import random
from consts import *
import numpy as np
import math
# from main import view_data as vd
import sys


class Weights:
    def __init__(self, weight: np.array, lead_layer, lag_layer):
        self.weight = weight  # 重み
        self.lag_layer = lag_layer  # Layer class object
        self.lead_layer = lead_layer  # Layer class object
        self.gradients = np.array([])
        self.moment = 0
        self.mt = 0
        self.vt = 0
        self.t = 1

    def calc_grad(self):
        self.gradients = np.dot(self.lag_layer.delta.T, self.lead_layer.activated_values).T / BATCH_SIZE

    def sgd(self):
        self.weight = self.weight - (ETA * self.gradients)

    def momentum_sgd(self):
        delta = (ETA * self.gradients) + self.moment
        self.weight = self.weight - delta
        self.moment = delta * BETA1

    def adam(self):
        self.mt = BETA1 * self.mt + ((1 - BETA1) * self.gradients)
        self.vt = BETA2 * self.vt + ((1 - BETA2) * np.power(self.gradients, 2))
        m = self.mt/(1 - pow(BETA1, self.t))
        v = self.vt/(1 - pow(BETA2, self.t))
        self.weight = self.weight - ALPHA * (m / (np.sqrt(v) + EPS))
        self.t += 1


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
        sd = []
        ave = []
        for v in self.net_values.T:
            sd.append(float(np.sqrt(np.var(v) + 0.0001)))  # avoid 0 divide
            ave.append(float(np.average(v)))
        self.sd = np.array(sd)
        self.net_values = (self.net_values - np.array(ave)) / self.sd

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = np.where(self.net_values < 0, 0, self.net_values/10)
        self.activated_values = np.vstack((self.activated_values.T, np.ones(batch_size))).T

    def calc_net_values(self):
        self.lag_weights.lag_layer.net_values = np.dot(self.activated_values, self.lag_weights.weight)

    def calc_delta(self, labels: list):
        norm_diff = (1/(BATCH_SIZE * self.sd))*(BATCH_SIZE - (self.net_values ** 2) - 1)
        round_z = self.lag_weights.lag_layer.delta.dot(self.lag_weights.weight[:-1].T) * norm_diff
        self.delta = round_z * np.where(self.net_values <= 0, 0, 1/10)


class SigmoidLayer(Layer):
    def normalize(self):
        pass

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = 1/(1 + np.exp(-self.net_values))
        self.activated_values = np.vstack((self.activated_values.T, np.ones(batch_size))).T

    def calc_delta(self, labels: list):
        round_z = np.dot(self.lag_weights.lag_layer.delta, self.lag_weights.weight[:-1].T)
        self.delta = round_z * (1 - self.activated_values.T[:-1].T) * self.activated_values.T[:-1].T


class SoftMaxLayer(Layer):
    def normalize(self):
        sd = []
        ave = []
        for v in self.net_values.T:
            sd.append(float(np.sqrt(np.var(v) + 0.0001)))  # avoid 0 divide
            ave.append(float(np.average(v)))
        self.sd = np.array(sd)
        self.net_values = (self.net_values - np.array(ave)) / self.sd

    def activate(self):
        value_arr = []
        for values in self.net_values:
            e = np.exp(values - values.max())
            value_arr.append((e / np.sum(e)).tolist())
        self.activated_values = np.array(value_arr)

    def calc_delta(self, labels: list) -> np.array:
        norm_diff = (1/(BATCH_SIZE * self.sd))*(BATCH_SIZE - pow(self.net_values, 2) - 1)
        delta_arr = np.array(self.activated_values.tolist())  # avoid_update_activated_value
        for i, delta in zip(labels, delta_arr):
            delta[i] = delta[i] - 1
        self.delta = np.array(delta_arr) * norm_diff

    def calc_net_values(self):
        pass


class NetWork:
    def __init__(self, hidden_layer: int, input_dim: int, activation: int, optimizer: int):
        self.hidden_layer = hidden_layer
        self.activation = activation
        self.optimizer = optimizer
        self.layers = []
        if self.activation == SIGMOID:
            self.layers.append(SigmoidLayer(input_dim))
            for i in range(hidden_layer - 1):
                self.layers.append(SigmoidLayer(MD1))
            self.layers.append(SigmoidLayer(MD2))
        if self.activation == ReLU:
            self.layers.append(Layer(input_dim))
            for i in range(hidden_layer - 1):
                self.layers.append(Layer(MD1))
            self.layers.append(Layer(MD2))

        self.layers.append(SoftMaxLayer(CLASS_NUM))
        self.weights = []
        self.__init_weight()
        self.data = None
        self.labels = None
        self.last_err = None
        self.previous_weights = None

    def __init_weight(self, i: int = 0):
        for lead, lag in zip(self.layers[i:-1], self.layers[i + 1:]):
            arr = np.random.randn(lead.node_amount + 1, lag.node_amount) / np.sqrt(lead.node_amount + 1)
            arr.T[-1] = 0.1
            weight = Weights(arr, lead, lag)
            lead.lag_weights = weight
            lag.lead_weights = weight
            self.weights.append(weight)
        self.weights[-1].weight.T[-1] = 0
        self.layers[-1].lead_weights.weight = self.layers[-1].lead_weights.weight * np.sqrt(2.0)

    def training(self, train_data, labels, train_layer_num=None):
        self.data = train_data
        self.labels = labels
        if not train_layer_num:
            index = len(self.weights)
        else:
            index = train_layer_num

        for item, key in zip(self.data, self.labels):
            # input_data_to_input_layer
            self.layers[0].net_values = np.array(item)
            # forward operation
            for layer in self.layers:
                layer.normalize()
                layer.activate()
                layer.calc_net_values()
            # back propagation
            for layer in list(reversed(self.layers))[:index]:
                layer.calc_delta(key)
            if self.optimizer == SGD:
                for weight in self.weights[-index:]:
                    weight.calc_grad()
                    weight.sgd()
            if self.optimizer == MomentumSGD:
                for weight in self.weights[-index:]:
                    weight.calc_grad()
                    weight.momentum_sgd()
            if self.optimizer == Adam:
                for weight in self.weights[-index:]:
                    weight.calc_grad()
                    weight.adam()

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
        return accuracy, error_average, self.l1_norm()/len(self.weights), self.l2_norm()/len(self.weights)

    def is_proved(self, err: list) -> bool:
        if not err:
            return True
        if not self.last_err:
            self.last_err = err[-1]
            return True
        if (self.last_err - err[-1]) / self.last_err > 0.01:
            self.last_err = err[-1]
            return True
        return False

    def add_layer(self):
        """
        重み情報を保存し層を追加して重みの初期化をする
        """
        self.previous_weights = self.weights
        self.layers = self.layers[:-2]
        if self.activation == SIGMOID:
            self.layers.extend([SigmoidLayer(MD1), SigmoidLayer(MD2), SoftMaxLayer(CLASS_NUM)])
        if self.activation == ReLU:
            self.layers.extend([Layer(MD1), Layer(MD2), SoftMaxLayer(CLASS_NUM)])
        self.weights = []
        self.__init_weight(len(self.weights))
        for target, previous in zip(self.weights[:-3], self.previous_weights):
            target.weight = previous.weight

    def rollback_layer(self):
        layers = []
        if self.activation == SIGMOID:
            layers.append(SigmoidLayer(self.hidden_layer))
            for i in range(len(self.layers)-4):
                layers.append(SigmoidLayer(MD1))
            layers.append(SigmoidLayer(MD2))
        if self.activation == ReLU:
            layers.append(Layer(self.hidden_layer))
            for i in range(len(self.layers)-4):
                layers.append(Layer(MD1))
            layers.append(Layer(MD2))
        layers.append(SoftMaxLayer(CLASS_NUM))
        self.layers = layers
        self.weights = []
        self.__init_weight()
        for target, previous in zip(self.weights, self.previous_weights):
            target.weight = previous.weight

    def l1_norm(self) -> float:
        norm = 0
        for weight in self.weights:
            norm += np.abs(weight.weight).sum()
        return norm

    def l2_norm(self) -> float:
        norm = 0
        for weight in self.weights:
            norm += (weight.weight ** 2).sum()
        return norm
