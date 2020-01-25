import csv
import os
from consts import *
import time
from classes import *
import numpy as np
import math
import random
import sys
import datetime


def main(model, dataset):
    # kaggle 用なので、trainデータを分割して使って性能評価する
    (train_datas, train_labels, test_datas, test_labels) = dataset
    train_data = train_datas[:SAMPLE_SIZE]
    train_label = train_labels[:SAMPLE_SIZE]
    # train_train or not
    test_data = train_datas[SAMPLE_SIZE:]
    test_label = train_labels[SAMPLE_SIZE:]
    print(len(test_label))
    print(test_data.shape)
    latest_epoch = 0
    c = 0
    training_information_dict = \
        {"train_accuracy": [], "train_error": [], "accuracy": [], "error": [], "l1_norm": [], "l2_norm": [],
         "node_amount": [], "epoch_time": [], "total_time": [], "weight_active_ratio": []}
    start = 0
    is_proved = True
    while is_proved:
        while not early_stopping(training_information_dict["error"], start):
            train_info = train_epoch(model, train_data, train_label)
            test_info = model.test(test_data, test_label)
            training_information_dict = save_info(train_info, test_info, training_information_dict)
            latest_epoch = model.propose_method(training_information_dict["accuracy"], latest_epoch)
            c += 1
        if model.is_proved(training_information_dict["accuracy"]):
            model.add_layer()
            start = c
        else:
            is_proved = False
    # additional leaning
    if model.is_dynamic:
        model.rollback_layer()
        for i in range(10):
            train_info = train_epoch(model, train_data, train_label, 2)
            test_info = model.test(test_data, test_label)
            training_information_dict = save_info(train_info, test_info, training_information_dict)
            c += 1
        for i in range(10):
            train_info = train_epoch(model, train_data, train_label, 1)
            test_info = model.test(test_data, test_label)
            training_information_dict = save_info(train_info, test_info, training_information_dict)
            c += 1
    print("train_end, total_time: {0}".format(training_information_dict["total_time"][-1]))
    return training_information_dict["accuracy"][-1]


def read_mnist() -> tuple:
    """
    mnistのデータセットを読み込んで配列に格納する
    データは1文字ごとに分けるため2次元配列、ラベルは1次元になる
    :return: train_data, train_label, test_data, test_labelのtuple
    """
    train_arr = np.array([i for i in csv.reader(open(os.path.join(DATA_DIR, MNIST[FileName][0])))][1:], dtype=int)
    train_label = train_arr.T[0].tolist()
    train_data = train_arr.T[1:].T
    print(len(train_label))
    print(train_data.shape)
    

    test_data = [i for i in csv.reader(open(os.path.join(DATA_DIR, MNIST[FileName][1])))][1:]

    return train_data, train_label, test_data, None

def transform_data(train_data: np.array, train_label: np.array) -> tuple:
    """
    データを3次元配列に整形する np.array([[[],[],[]],[[],[],[]]]) 的なイメージ
    :param train_data: 学習データの2次元配列
    :param train_label: 学習データのラベル
    :return: 整形後のデータとラベルのタプル
    """
    dim = int(train_data[0].size)
    batch_num = int(train_data.size/(BATCH_SIZE * dim))
    return_data = train_data.reshape((batch_num, BATCH_SIZE, dim))
    return_label = train_label.reshape((batch_num, BATCH_SIZE)).tolist()
    return return_data, return_label


def shuffle_data(data: np.array, label: list) -> tuple:
    """
    受け取ったデータ配列とラベル配列の順番をランダムに入れ替える
    :param data: データの2次元配列
    :param label: ラベルの配列
    :return: ランダムに入れ替えたtrain_dataとそれに対応したtrain_label
    """
    array = np.vstack((data.T, label)).T.tolist()
    array = np.array(random.sample(array, SAMPLE_SIZE))
    return_data = array.T[:-1].T
    return_label = array.T[-1]

    return return_data, return_label


def train_epoch(network, train_data, train_label, train_layer_num=None):
    data, label = transform_data(*shuffle_data(train_data, train_label))
    start = time.time()
    if not train_layer_num:
        accuracy, error, weight_active_ratio = network.training(data, label)
    else:
        accuracy, error, weight_active_ratio = network.training(data, label, train_layer_num)
    end = time.time()
    return end - start, accuracy, error, weight_active_ratio


def save_info(train_info, test_info, training_information_dict):
    """
    dict have...
    "train_accuracy": [], "train_error": [], "accuracy": [], "error": [], "l1_norm": [],
    "l2_norm": [], "node_amount": [], "epoch_time": [], "total_time": [], "weight_active_ratio" :[]
    train info have...
    (epoch_time, train_accuracy, train_error, weight_active_ratio)
    test_info have...
    (accuracy, error, l1_norm, l2_norm, node_amount)
    """
    return_dict = copy.deepcopy(training_information_dict)
    return_dict["epoch_time"].append(train_info[0])
    return_dict["total_time"].append(sum(return_dict["epoch_time"]))
    return_dict["train_accuracy"].append(train_info[1])
    return_dict["train_error"].append(train_info[2])
    return_dict["weight_active_ratio"].append(train_info[3])
    return_dict["accuracy"].append(test_info[0])
    return_dict["error"].append(test_info[1])
    return_dict["l1_norm"].append(test_info[2])
    return_dict["l2_norm"].append(test_info[3])
    return_dict["node_amount"].append(test_info[4])
    return return_dict


def early_stopping(err: list, start) -> bool:
    if len(err) == start:
        return False
    arg_min = np.argmin(np.array(err[start:]))
    if (len(err[start:]) - arg_min) < EARLY_STOPPING_EPOCH:
        return False
    return True


if __name__ == "__main__":
    data_tuple = read_mnist()
    st = NetWork(hidden_layer=2, in_dim=MNIST[DataLength], activation=ReLU, optimizer=ADAM,
                 md1=50, md2=100, out_dim=CLASS_NUM, dynamic=False, propose=False)
    dy = NetWork(hidden_layer=2, in_dim=MNIST[DataLength], activation=ReLU, optimizer=ADAM,
                 md1=50, md2=100, out_dim=CLASS_NUM, dynamic=True, propose=False)
    pr = NetWork(hidden_layer=2, in_dim=MNIST[DataLength], activation=ReLU, optimizer=ADAM,
                 md1=50, md2=100, out_dim=CLASS_NUM, dynamic=True, propose=True)
    st_accuracy = main(st, data_tuple)
    print(st_accuracy)
    dy_accuracy = main(dy, data_tuple)
    print(dy_accuracy)
    pr_accuracy = main(pr, data_tuple)
    print(pr_accuracy)
