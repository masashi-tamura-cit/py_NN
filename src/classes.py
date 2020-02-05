from consts import *
import numpy as np
import math
import copy
import sys


class Sgd:
    name = "SGD"

    def optimize(self, weight, grad):
        return weight - (ETA * grad)

    def reset(self):
        pass


class MomentumSgd(Sgd):
    name = "Momentum_SGD"

    def __init__(self):
        self.moment = 0

    def optimize(self, weight, grad):
        delta = (ETA * grad) + self.moment
        self.moment = delta * BETA1
        return weight - delta

    def reset(self):
        self.moment = 0


class Adam(Sgd):
    name = "Adam"

    def __init__(self):
        self.mt = 0
        self.vt = 0
        self.__t = 1

    def optimize(self, weight, grad):
        self.mt = (BETA1 * self.mt) + ((1 - BETA1) * grad)
        self.vt = (BETA2 * self.vt) + ((1 - BETA2) * np.power(grad, 2))
        m = self.mt/(1 - pow(BETA1, self.__t))
        v = self.vt/(1 - pow(BETA2, self.__t))
        self.__t += 1
        return weight - ETA * (m / (np.sqrt(v) + EPS))

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
        self.previous_weights = weight

    def calc_grad(self, lm: float = LAMBDA):
        if self.is_fixed:
            return None
        self.gradients = (np.dot(self.lag_layer.delta.T, self.lead_layer.activated_values).T +
                          (lm * np.sign(self.weight))) / BATCH_SIZE

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
        border = np.sort(np.abs(self.weight[np.nonzero(self.weight)].flat))[deactive_amount]
        self.active_set = np.where(np.abs(self.weight) < border, 0, 1)
        self.update_weights()

    def update_weights(self):
        self.previous_weights = self.weight
        self.weight = self.weight * self.active_set

    def rollback_weight(self):
        self.weight = self.previous_weights


class Layer: 
    def __init__(self, amount: int):
        self.node_amount = amount
        self.net_values = np.array([])
        self.activated_values = np.array([])
        self.lead_weights = None
        self.lag_weights = None
        self.delta = np.array([])
        self.test_nodes = np.array([])
        self.var = []
        self.active_set = np.ones(amount)
        self.ave_ave = np.zeros(amount)
        self.var_ave = np.zeros(amount)
        self.gamma = np.ones(amount)
        self.beta = np.zeros(amount)

    def normalize(self):
        self.var = np.var(self.net_values, axis=0)
        ave = np.average(self.net_values, axis=0)
        self.net_values = self.gamma * ((self.net_values - ave) / np.sqrt(self.var + EPS)) + self.beta
        self.ave_ave = (MOMENTUM * self.ave_ave) + ((1 - MOMENTUM) * ave)
        self.var_ave = (MOMENTUM * self.var_ave) + ((1 - MOMENTUM) * self.var)

    def test_normalize(self):
        self.net_values = self.gamma * ((self.net_values - self.ave_ave) / np.sqrt(self.var_ave + EPS)) + self.beta

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = self.net_values * (self.net_values > 0)
        self.activated_values = np.vstack((self.activated_values.T, np.ones(batch_size))).T

    def calc_net_values(self):
        self.lag_weights.lag_layer.net_values \
            = np.dot(self.activated_values, self.lag_weights.weight)

    def calc_delta(self, labels: list):
        round_z = self.lag_weights.lag_layer.delta.dot(self.lag_weights.weight[:-1].T)
        round_net = ((self.net_values * (self.net_values > 0)) / self.net_values) * round_z
        x_hat = (self.net_values - self.beta) / self.gamma
        delta_gamma = np.sum(round_net * x_hat, axis=0)
        delta_beta = np.sum(round_net, axis=0)
        self.delta = (self.gamma/(np.sqrt(self.var + EPS))) * \
                     (round_net - (1 / BATCH_SIZE) *
                      (delta_beta + (x_hat * delta_gamma)))

        # SGD gamma and beta
        self.gamma = self.gamma - (ETA * delta_gamma / BATCH_SIZE)
        self.beta = self.beta - (ETA * delta_beta / BATCH_SIZE)

    def __make_active_set(self):
        """
        active set を作る
        0，1の１次元配列で0が非active, 1がactiveとなる
        L1ノルムが小さい（平均 - 標準偏差の2倍以下）　OR 類似度が大きい(0.85)ratio割のノードを非activeにする
        :param ratio: 非active化する割合
        :return: None
        """
        # lead_weight, lag_weightの集計
        lead_weight = self.lead_weights.weight
        lead_weight_l1 = np.sum(np.abs(self.lead_weights.weight * self.lead_weights.active_set), axis=0)
        lag_weight_l1 = np.sum(np.abs(self.lag_weights.weight * self.lag_weights.active_set), axis=1)[:-1]
        deactivate_priority = lead_weight_l1 + lag_weight_l1
        ave = np.average(deactivate_priority)
        sd = np.var(deactivate_priority) ** .5

        # cos similarity
        norm = (lead_weight ** 2).sum(0, keepdims=True) ** .5
        cos_similarity = lead_weight.T @ lead_weight / norm / norm.T
        max_similarity = [np.max(cos_similarity[i][i + 1:]) for i in range(cos_similarity.shape[0] - 1)]
        max_similarity.append(0)

        # make_active_set
        border = ave - (1.5 * sd)
        self.active_set = np.where(deactivate_priority < border, 0, 1) * np.where(np.array(max_similarity) > 0.8, 0, 1)
        print(f"{self.node_amount - sum(np.where(deactivate_priority < border, 0, 1))} norm_base")
        print(f"{self.node_amount - sum(np.where(np.array(max_similarity) > 0.8, 0, 1))} similarity_base")

    def delete_node(self, optimizer):
        """
        非activeなノードを削除する
        削除したノードに繋がる重みも列（行）を一括で削除する
        :return: 削除後のノード数
        """
        # net_value削除
        # lag_weightの該当列削除
        # lead_weightの該当行削除
        # gamma, betaの削除
        c = 0
        self.__make_active_set()
        for i in range(self.active_set.size):
            if self.active_set[i] == 0:
                # delete linked weights
                self.lead_weights.weight = np.delete(self.lead_weights.weight, i - c, axis=1)
                self.lead_weights.active_set = np.delete(self.lead_weights.active_set, i - c, axis=1)
                self.lag_weights.weight = np.delete(self.lag_weights.weight, i - c, axis=0)
                self.lag_weights.active_set = np.delete(self.lag_weights.active_set, i - c, axis=0)
                # delete self params
                self.gamma = np.delete(self.gamma, i-c)
                self.beta = np.delete(self.beta, i-c)
                self.ave_ave = np.delete(self.ave_ave, i-c)
                self.var_ave = np.delete(self.var_ave, i-c)
                c += 1
                # delete optimizer params
                if isinstance(optimizer, Adam):
                    self.lead_weights.optimizer.mt = np.delete(self.lead_weights.optimizer.mt, i - c, axis=1)
                    self.lead_weights.optimizer.vt = np.delete(self.lead_weights.optimizer.vt, i - c, axis=1)
                    self.lag_weights.optimizer.mt = np.delete(self.lag_weights.optimizer.mt, i - c, axis=0)
                    self.lag_weights.optimizer.vt = np.delete(self.lag_weights.optimizer.vt, i - c, axis=0)
                if isinstance(optimizer, MomentumSgd):
                    self.lead_weights.optimizer.moment = np.delete(self.lead_weights.optimizer.moment, i - c, axis=1)
                    self.lag_weights.optimizer.moment = np.delete(self.lag_weights.optimizer.moment, i - c, axis=0)
        # set node_amount
        self.node_amount = np.sum(self.active_set)
        self.active_set = np.ones(self.node_amount)
        return self.node_amount


class SigmoidLayer(Layer):

    def activate(self):
        batch_size = self.net_values.shape[0]
        self.activated_values = 1/(1 + np.exp(-self.net_values))
        self.activated_values = np.vstack((self.activated_values.T, np.ones(batch_size))).T

    def calc_delta(self, labels: list):
        round_z = self.lag_weights.lag_layer.delta.dot(self.lag_weights.weight[:-1].T)
        round_net = (1 - self.activated_values.T[:-1].T) * round_z
        x_hat = (self.net_values - self.beta) / self.gamma
        delta_gamma = np.sum(round_net * x_hat, axis=0)
        delta_beta = np.sum(round_net, axis=0)
        self.delta = (self.gamma/(np.sqrt(self.var + EPS))) * \
                     (round_net - (1 / BATCH_SIZE) *
                      (delta_beta + (x_hat * delta_gamma)))

        # SGD gamma and beta
        self.gamma = self.gamma - (ETA * delta_gamma)
        self.beta = self.beta - (ETA * delta_beta)


class SoftMaxLayer(Layer):

    def activate(self):
        value_arr = []
        for values in self.net_values:
            e = np.exp(values - values.max())
            value_arr.append((e / np.sum(e)).tolist())
        self.activated_values = np.array(value_arr)

    def calc_delta(self, labels: list) -> np.array:
        round_net = copy.deepcopy(self.activated_values)
        for i, delta in zip(labels, round_net):
            delta[i] = delta[i] - 1
        x_hat = (self.net_values - self.beta) / self.gamma
        delta_gamma = np.sum(round_net * x_hat, axis=0)
        delta_beta = np.sum(round_net, axis=0)
        self.delta = (self.gamma/(np.sqrt(self.var + EPS))) * \
                     (round_net - (1 / BATCH_SIZE) *
                      (delta_beta + (x_hat * delta_gamma)))

        # SGD gamma and beta
        self.gamma = self.gamma - (ETA * delta_gamma)
        self.beta = self.beta - (ETA * delta_beta)

    def calc_net_values(self):
        pass


class NetWork:
    def __init__(self, hidden_layer: int, in_dim: int, activation: int,
                 optimizer: int, md1: int, md2: int, out_dim: int, dynamic=True, propose=True):
        self.hidden_layer = hidden_layer
        self.activation = activation
        self.layers = []
        self.is_dynamic = dynamic
        self.is_propose = propose
        self.layer_dims = [in_dim, md1, md2, out_dim]
        self.property_str = self.__set_property_str(in_dim, md1, md2, optimizer, activation)

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
        self.previous_info = {"weights": [], "layers": [], "layer_dims": []}
        self.deactivate_ratio = {"input_node": {"ratio": 0.1, "alfa": 0.9},
                                 "weight": {"ratio": 0.5, "alfa": 0.4},
                                 "node": {"ratio": 0.2, "alfa": 0.4}}  # target: [deactive_ratio, decline_ratio]
        self.threshold = 0.01

    def __set_property_str(self, in_dim, md1, md2, optimizer, activation):
        """
        self.hidden_layer
        self.in_dim
        self.is_dynamic
        self.is_propose
        あと入力情報からディレクトリ名に使う文字列を生成する
        :return: ディレクトリ名文字列
        e.g. MNIST_dynamic_propose_2_50_100_Adam0.01_ReLU
             CIFAR10_static__3_500_1000_Momentum0.1_Sigmoid
        """
        if optimizer == ADAM:
            opt = "Adam"
        elif optimizer == MOMENTUM_SGD:
            opt = "Momentum"
        else:
            opt = "SGD"
        data = "MNIST" if in_dim == 784 else "CIFAR10"
        activator = "ReLU" if activation == ReLU else "sigmoid"
        state = "dynamic"if self.is_dynamic else "static"
        propose = "propose" if self.is_propose else ""
        return "PRIME_{0}_{1}_{2}_{3}_{4}_{5}_{6}{7}_{8}".format(data, state, propose, self.hidden_layer,
                                                                 md1, md2, opt, ETA, activator)

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

    def training(self, train_data, labels, train_layer_num=None, lm=LAMBDA) -> tuple:
        error = 0
        accuracy = 0
        norm_c = lm if self.is_propose else 0
        if not train_layer_num:
            # index = 3
            index = len(self.weights)
        else:
            index = train_layer_num
        for item, key in zip(train_data, labels):
            # input_data_to_input_layer
            self.layers[0].net_values = np.array(item)
            # forward operation
            for layer, i in zip(self.layers, range(len(self.layers))):
                layer.normalize()
                layer.activate()
                layer.calc_net_values()
            error += self.calc_error_sum(key)
            accuracy += self.calc_correct_num(key)
            # back propagation
            for layer in list(reversed(self.layers))[:index]:
                layer.calc_delta(key)
            for weight in self.weights[-index:]:
                weight.calc_grad(lm=norm_c)
                weight.optimize()
        return accuracy/SAMPLE_SIZE, error / SAMPLE_SIZE, self.weight_active_percent()

    def test(self, test_data: np.array, test_labels: list) -> tuple:
        self.layers[0].net_values = test_data
        data_amount = len(test_labels)
        for layer in self.layers:
            layer.test_normalize()
            layer.activate()
            layer.calc_net_values()
        accuracy = self.calc_correct_num(test_labels) / data_amount
        error = self.calc_error_sum(test_labels) / data_amount
        return accuracy, error, self.l1_norm(), self.l2_norm(), self.total_node()

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
        if not self.is_dynamic:
            return None
        self.previous_info["weights"] = copy.deepcopy(self.weights)
        self.previous_info["layers"] = copy.deepcopy(self.layers)
        self.previous_info["layer_dims"] = copy.deepcopy(self.layer_dims)
        self.layers = self.layers[:-2]
        if self.activation == SIGMOID:
            self.layers.extend([SigmoidLayer(self.layer_dims[1]),
                                SigmoidLayer(self.layer_dims[-2]), SoftMaxLayer(self.layer_dims[-1])])
        if self.activation == ReLU:
            self.layers.extend([Layer(self.layer_dims[1]),
                                Layer(self.layer_dims[-2]), SoftMaxLayer(self.layer_dims[-1])])
        self.weights = []
        self.__init_weight(len(self.weights))
        for target, previous in zip(self.weights[:-3], self.previous_info["weights"]):
            target.weight = copy.deepcopy(previous.weight)
            target.is_fixed = True
        for w in self.weights:
            w.optimizer.reset()
        self.layer_dims = []
        for l in self.layers:
            self.layer_dims.append(l.node_amount)
            l.active_set = np.ones(l.node_amount)

    def rollback_layer(self):
        if not self.is_dynamic:
            return None
        self.layers = self.previous_info["layers"]
        self.layer_dims = self.previous_info["layer_dims"]
        self.weights = self.previous_info["weights"]
        for i, w in enumerate(self.weights):
            self.weights[i].lead_layer = self.layers[i]
            self.layers[i].lag_weight = self.weights[i]
            self.weights[i].lag_layer = self.layers[i + 1]
            self.layers[i + 1].lag_weight = self.weights[i]

    def l1_norm(self) -> float:
        norm = 0
        for weight in self.weights:
            norm += np.abs(weight.weight * weight.active_set).sum()
        return norm

    def l2_norm(self) -> float:
        norm = 0
        for weight in self.weights:
            norm += ((weight.weight * weight.active_set) ** 2).sum()
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
            error += -math.log(value[key]+EPS, np.e)
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

    def propose_method(self, accuracy_list: list, latest_epoch: int) -> int:
        """
        ノードと重みのスパース推定を行う
        入力から3つのオペレーションを行う
        １: 最後にスパース化してからまだ性能が向上しそうな場合：　何も処理せずにlatest_epochを返す
        2: 最後にスパース化してから、する前の性能以上の性能が出た場合： そのままさらにスパース化する
        3: 性能向上が頭打ちになって、スパース化する前の性能を下回りそうな場合： ロールバックしてレートを落とす
        """
        print("propose_method ")
        if not self.is_propose:
            return 0
        input_layer = self.layers[0]

        # operation_1 スパース化してからの最高の性能が出てから一定エポックが経過していない
        target = accuracy_list[latest_epoch:]
        if not target or len(target) - max(enumerate(target), key=lambda x: x[1])[0] < 5:
            return latest_epoch

        # 性能が向上したと認められない場合、ロールバックしてレートを落とす
        previous_accuracy = accuracy_list[:latest_epoch]
        if previous_accuracy:
            print(f"after:{max(target)} before:{max(previous_accuracy)}")
        if previous_accuracy and max(target) < max(previous_accuracy) + self.threshold:
            for w in self.weights:
                w.active_set = 1
                w.rollback_weight()
                self.deactivate_ratio["weight"]["ratio"] *= self.deactivate_ratio["weight"]["alfa"]
        # スパース化
        for i, l in enumerate(self.layers[-3:-1]):
            if not l.lead_weights is None:
                self.layer_dims[-3 + i] = l.delete_node(self.optimizer)
        for w in self.weights:
            w.make_active_set(self.deactivate_ratio["weight"]["ratio"])

        if previous_accuracy and max(target) >= max(previous_accuracy) + self.threshold:
            for i in self.layers[1:]:
                print(i.lead_weights.weight.shape)
                print(self.weight_active_percent())
        return len(accuracy_list) - 1

    def weight_active_percent(self):
        if not self.is_propose:
            return 1
        all_weight_amount = 0
        active_weight_amount = 0
        for w in self.weights:
            all_weight_amount += np.prod(w.weight.shape)
            active_weight_amount += np.nonzero(w.weight * w.active_set)[0].shape[0]
        return active_weight_amount / all_weight_amount
