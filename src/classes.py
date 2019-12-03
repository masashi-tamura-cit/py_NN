from consts import *
import numpy as np
import math
import copy
import sys
import time


class Sgd:
    def optimize(self, weight, grad):
        return weight - (ETA * grad)

    def reset(self):
        pass


class MomentumSgd(Sgd):
    def __init__(self):
        self.moment = 0

    def optimize(self, weight, grad):
        delta = (ETA * grad) + self.moment
        self.moment = delta * BETA1
        return weight - delta

    def reset(self):
        self.moment = 0


class Adam(Sgd):
    def __init__(self):
        self.mt = 0
        self.vt = 0
        self.__t = 1

    def optimize(self, weight, grad):
        self.mt = BETA1 * self.mt + ((1 - BETA1) * grad)
        self.vt = BETA2 * self.vt + ((1 - BETA2) * np.power(grad, 2))
        m = self.mt/(1 - pow(BETA1, self.__t))
        v = self.vt/(1 - pow(BETA2, self.__t))
        self.__t += 1
        return weight - ALPHA * (m / (np.sqrt(v) + EPS))

    def reset(self):
        self.mt = 0
        self.vt = 0
        self.__t = 1


class Weights:
    def __init__(self, weight: np.array, lead_layer, lag_layer, optimizer):
        self.weight = weight  # 重み
        self.lag_layer = lag_layer  # Layer class object
        self.lead_layer = lead_layer  # Layer class object
        self.gradients = np.array([])
        self.optimizer = optimizer
        self.is_fixed = False
        self.active_set = np.ones(weight.shape)
        self.previous_active_set = np.ones(weight.shape)

    def calc_grad(self):
        if self.is_fixed:
            return None

        self.gradients = (np.dot(self.lag_layer.delta.T, self.lead_layer.activated_values).T +
                          (LAMBDA * np.sign(self.weight))) / BATCH_SIZE

    def optimize(self):
        if self.is_fixed:
            return None
        self.weight = self.optimizer.optimize(self.weight, self.gradients)

    def make_active_set(self, ratio):
        """
        重みのactive_set行列（２次元）を作る
        重みの絶対値が０に近いものから順に非activeにする
        :param ratio: あらたに非activeにする重みの数の全体のactiveな重み本数に対する割合
        :return: None
        """
        deactive_amount = int(np.sum(self.active_set) * ratio)
        if deactive_amount <= 1:
            return None
        self.previous_active_set = self.active_set
        border = np.sort(np.abs(self.weight.flat))[deactive_amount]
        self.active_set = np.where(np.abs(self.weight) < border, 0, 1)
        print(self.weight.shape)

class Layer: 
    def __init__(self, amount: int):
        self.node_amount = amount
        self.net_values = np.array([])
        self.activated_values = np.array([])
        self.lead_weights = None
        self.lag_weights = None
        self.delta = np.array([])
        self.test_nodes = np.array([])
        self.sd = []
        self.active_set = np.ones(amount)

    def normalize(self):
        self.sd = np.sqrt(np.var(self.net_values, axis=0) + EPS)
        ave = np.average(self.net_values, axis=0)
        self.net_values = (self.net_values - ave) / (self.sd)

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = self.net_values * (self.net_values > 0)
        self.activated_values = np.vstack((self.activated_values.T, np.ones(batch_size))).T

    def calc_net_values(self):
        self.lag_weights.lag_layer.net_values \
            = np.dot(self.activated_values, (self.lag_weights.weight * self.lag_weights.active_set)) * self.lag_weights.lag_layer.active_set + EPS

    def calc_delta(self, labels: list):
        norm_diff = (1/(BATCH_SIZE * self.sd))*(BATCH_SIZE - (self.net_values ** 2) - 1)
        round_z = self.lag_weights.lag_layer.delta.dot(self.lag_weights.weight[:-1].T) * norm_diff
        self.delta = round_z * (self.net_values * (self.net_values > 0))/ self.net_values

    def make_active_set(self, ratio):
        """
        active set を作る
        ０，nの１次元配列で０が非active, 1がactiveとなる
        L1ノルム * 類似度が大きいratio割のノードを非activeにする
        他の重みが消える分、生き残った重みは均等にall/active倍にする
        :param ratio: 非active化する割合
        :return: None
        """
        # lead_weightの集計
        # lag_weightの集計
        # active_set 1次元配列の作成

        current_active_magnification = np.max(self.active_set)
        active_amount = int(self.node_amount - (self.node_amount * ratio))
        if self.node_amount - active_amount <= 1:
            return None
        if self.lead_weights is None:
            return None
        lead_weight = self.lead_weights.weight
        lead_weight_l1 = np.sum(np.abs(self.lead_weights.weight * self.lead_weights.active_set), axis=0)
        lag_weight_l1 = np.sum(np.abs(self.lag_weights.weight * self.lag_weights.active_set), axis=1)[:-1]
        weights_l1_nrom_sum = lead_weight_l1 + lag_weight_l1
        norm = (lead_weight ** 2).sum(0, keepdims = True) ** .5
        cos_similarity =1 - (lead_weight.T @ lead_weight/ norm / norm.T)
        max_similarity = [np.min(np.delete(cos_similarity[i], [i])) for i in range(cos_similarity.shape[0])]
        similarity = np.abs(np.log(max_similarity))
        deactivate_priority = weights_l1_nrom_sum * similarity
        border = np.sort(deactivate_priority.flat)[active_amount]
        self.active_set = np.where(deactivate_priority > border,0, 1)
        self.active_set = self.active_set * current_active_magnification * (self.active_set.size / np.sum(self.active_set))

    def delete_node(self, optimizer):
        """
        非activeなノードを削除する
        削除したノードに繋がる重みも列（行）を一括で削除する
        :return: 削除後のノード数
        """
        # net_value削除
        # lag_weightの該当列削除
        # lead_weightの該当行削除
        c = 0
        for i in range(self.active_set.size):
            if self.active_set[i] == 0:
                self.lead_weights.weight = np.delete(self.lead_weights.weight, i - c, axis=1)
                self.lead_weights.active_set = np.delete(self.lead_weights.active_set, i - c, axis=1)
                self.lag_weights.weight = np.delete(self.lag_weights.weight,i - c, axis=0)
                self.lag_weights.active_set = np.delete(self.lag_weights.active_set,i - c, axis=0)
                c += 1
                if isinstance(optimizer, Adam):
                    self.lead_weights.optimizer.mt = np.delete(self.lead_weights.optimizer.mt, i - c, axis=1)
                    self.lead_weights.optimizer.vt = np.delete(self.lead_weights.optimizer.vt, i - c, axis=1)
                    self.lag_weights.optimizer.mt = np.delete(self.lag_weights.optimizer.mt,i - c, axis=0)
                    self.lag_weights.optimizer.vt = np.delete(self.lag_weights.optimizer.vt,i - c, axis=0)
                if isinstance(optimizer, MomentumSgd):
                    self.lead_weights.optimizer.moment= np.delete(self.lead_weights.optimizer.moment, i - c, axis=1)
                    self.lag_weights.optimizer.moment= np.delete(self.lag_weights.optimizer.moment, i - c, axis=0)
        self.node_amount = int(round(np.sum(self.active_set)/np.max(self.active_set.flat)))
        print(self.node_amount)
        return self.node_amount

class SigmoidLayer(Layer):

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = 1/(1 + np.exp(-self.net_values))
        self.activated_values = np.vstack((self.activated_values.T, np.ones(batch_size))).T

    def calc_delta(self, labels: list):
        norm_diff = (1/(BATCH_SIZE * self.sd))*(BATCH_SIZE - (self.net_values ** 2) - 1)
        round_z = np.dot(self.lag_weights.lag_layer.delta, self.lag_weights.weight[:-1].T)
        self.delta = round_z * (1 - self.activated_values.T[:-1].T) * self.activated_values.T[:-1].T * norm_diff


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
        delta_arr = copy.deepcopy(self.activated_values)
        for i, delta in zip(labels, delta_arr):
            delta[i] = delta[i] - 1
        self.delta = np.array(delta_arr) * norm_diff

    def calc_net_values(self):
        pass


class NetWork:
    def __init__(self, hidden_layer: int, in_dim: int, activation: int, optimizer: int, md1: int, md2: int, out_dim: int):
        self.hidden_layer = hidden_layer
        self.activation = activation
        self.layers = []
        self.layer_dims = [in_dim, md1, md2, out_dim]

        # set layers
        if activation == SIGMOID:
            self.layers.append(SigmoidLayer(in_dim))
            for i in range(hidden_layer - 1):
                self.layers.append(SigmoidLayer(md1))
            self.layers.append(SigmoidLayer(md2))
        elif activation == ReLU:
            self.layers.append(Layer(in_dim))
            for i in range(hidden_layer - 1):
                self.layers.append(Layer(md1))
            self.layers.append(Layer(md2))
        else:
            print("bad input: activation_func")
            sys.exit()
        self.layers.append(SoftMaxLayer(out_dim))

        # set optimizer
        if optimizer == ADAM:
            self.optimizer = Adam()
        elif optimizer == MOMENTUM_SGD:
            self.optimizer = MomentumSgd()
        elif optimizer == SGD:
            self.optimizer = Sgd()
        else:
            print("bad input: optimizer")
            sys.exit()

        self.weights = []
        self.__init_weight()
        self.last_accuracy = None
        self.previous_weights = None
        self.deactivate_ratio = {"input_node": {"ratio": 0.1,"alfa": 0.9}, 
                                 "weight": {"ratio": 0.2,"alfa": 0.9}, 
                                 "node": {"ratio": 0.4,"alfa": 0.5}}  # target: [deactive_ratio, decline_ratio]

    def __init_weight(self, i: int = 0):
        for lead, lag in zip(self.layers[i:-1], self.layers[i + 1:]):
            arr = np.random.randn(lead.node_amount + 1, lag.node_amount) * (np.sqrt(2.0 / (lead.node_amount + 1)))
            arr.T[-1] = 0.1
            weight = Weights(arr, lead, lag, copy.deepcopy(self.optimizer))
            lead.lag_weights = weight
            lag.lead_weights = weight
            self.weights.append(weight)
        self.weights[-1].weight.T[-1] = 0
        self.layers[-1].lead_weights.weight = self.layers[-1].lead_weights.weight / np.sqrt(2.0)

    def training(self, train_data, labels, train_layer_num=None) -> tuple:
        error = 0
        accuracy = 0
        if not train_layer_num:
            index = 3
        else:
            index = train_layer_num
        for item, key in zip(train_data, labels):
            # input_data_to_input_layer
            self.layers[0].net_values = np.array(item)
            # forward operation
            for layer in self.layers:
                layer.normalize()
                layer.activate()
                layer.calc_net_values()
            error += self.calc_error_sum(key)
            accuracy += self.calc_correct_num(key)
            # back propagation
            for layer in list(reversed(self.layers))[:index]:
                layer.calc_delta(key)
            for weight in self.weights[-index:]:
                weight.calc_grad()
                weight.optimize()
        return accuracy/SAMPLE_SIZE, error/SAMPLE_SIZE

    def test(self, test_data: np.array, test_labels: list) -> tuple:
        self.layers[0].net_values = test_data
        data_amount = len(test_labels)
        for layer in self.layers:
            layer.normalize()
            layer.activate()
            layer.calc_net_values()
        accuracy = self.calc_correct_num(test_labels) / data_amount
        error_average = self.calc_error_sum(test_labels) / data_amount
        return accuracy, error_average, self.l1_norm(), self.l2_norm(), self.total_node()

    def is_proved(self, accuracy: list) -> bool:
        if not accuracy:
            return True
        value = max(accuracy[-EARLY_STOPPING_EPOCH:])
        if not self.last_accuracy:
            self.last_accuracy = value
            return True
        if value - self.last_accuracy >= 0.01:
            self.last_accuracy = value
            return True
        return False

    def add_layer(self):
        """
        重み情報を保存し層を追加して重みの初期化をする
        """
        self.previous_weights = self.weights
        self.layers = self.layers[:-2]
        if self.activation == SIGMOID:
            self.layers.extend([SigmoidLayer(self.layer_dims[1]), SigmoidLayer(self.layer_dims[-2]), SoftMaxLayer(self.layer_dims[-1])])
        if self.activation == ReLU:
            self.layers.extend([Layer(self.layer_dims[1]), Layer(self.layer_dims[-2]), SoftMaxLayer(self.layer_dims[-1])])
        self.weights = []
        self.__init_weight(len(self.weights))
        for target, previous in zip(self.weights[:-3], self.previous_weights):
            target.weight = previous.weight
            target.is_fixed = True
        for w in self.weights:
            w.optimizer.reset()

    def rollback_layer(self):
        layers = []
        if self.activation == SIGMOID:
            layers.append(SigmoidLayer(self.layer_dims[0]))
            for i in range(len(self.layers[1:-2])):
                layers.append(SigmoidLayer(self.layer_dims[1]))
            layers.append(SigmoidLayer(self.layer_dims[-2]))
        if self.activation == ReLU:
            layers.append(Layer(self.layer_dims[0]))
            for i in range(len(self.layers[1:-2])):
                layers.append(Layer(self.layer_dims[1]))
            layers.append(Layer(self.layer_dims[-2]))
        layers.append(SoftMaxLayer(self.layer_dims[-1]))
        self.layers = layers
        seweights = []
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

    def total_node(self) -> int:
        node_amount = 0
        for layer in self.layers:
            node_amount += layer.node_amount
        return node_amount

    def calc_error_sum(self, label: list) -> float:
        """
        1バッチの誤差関数値合計を返す関数
        :param label: 入力データのラベル
        :return: 誤差関数値合計
        """
        error = 0
        for value, key in zip(self.layers[-1].activated_values, label):
            error += -math.log(value[key], np.e)
        return error

    def calc_correct_num(self, label: list) -> int:
        """
        １バッチの正解数を返す関数
        :param label: 入力データのラベル
        :return: 正解数
        """
        acc = 0
        for value, key in zip(self.layers[-1].activated_values, label):
            if np.argmax(value) == key:
                acc += 1
        return acc

    def propose_method(self, prev_error, now_error) -> None:
        """
        ノードと重みのスパース推定を行う
        """
        input_layer = self.layers[0]
        # check is_not_decline
        if prev_error - now_error < 0.1 and self.layers[1].active_set.size == np.sum(self.layers[1].active_set):
        # rollback and ratio decliment
            if len(self.layers) == 4:
                input_layer.active_set = np.ones(input_layer.node_amount)
                self.deactivate_ratio["input_node"]["ratio"] *= self.deactivate_ratio["input_node"]["alfa"]
            for l in self.layers[-3:-1]:
                l.active_set = np.ones(l.node_amount)
                self.deactivate_ratio["node"]["ratio"] *= self.deactivate_ratio["node"]["alfa"]
            for w in self.weights:
                w.active_set = w.previous_active_set
                self.deactivate_ratio["weight"]["ratio"] *= self.deactivate_ratio["weight"]["alfa"]
        print("sparse")

        # if len(self.layers) == 4:
        #    node_amount =  input_layer.delete_node(None)
        #    self.layer_dims[1] = node_amount
        #    input_layer.make_active_set(self.deactivate_ratio["input_node"]["ratio"])
        for l, d in zip(self.layers[-3:-1],self.layer_dims[-3:-1]):
            if not l.lead_weights is None:
                node_amount = l.delete_node(self.optimizer)
                l.make_active_set(self.deactivate_ratio["node"]["ratio"])

        for w in self.weights:
            w.make_active_set(self.deactivate_ratio["weight"]["ratio"])
