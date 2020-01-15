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
    (train_data, train_label, test_data, test_label) = dataset
    train_data = train_data[:SAMPLE_SIZE]
    train_label = train_label[:SAMPLE_SIZE]
    # train_train or not
    test_data = test_data[:VALIDATION_DATA]
    test_label = test_label[:VALIDATION_DATA]
    # test_data = train_data[:VALIDATION_DATA]
    # test_label = train_label[:VALIDATION_DATA]
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
    make_csv(training_information_dict, model.property_str)


def read_cifar10() -> tuple:
    """
    CIFAR10のデータセットを読み込んで配列に格納する
    データは1画像毎に分けるため2次元配列、ラベルは1次元
    :return: train_data, train_label, test_data, test_label
    """
    bin_data = []
    # read_train_data
    for filename in CIFAR10[FileName][:-1]:
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, "rb") as f:
            bin_data.extend(f.read((CIFAR10[DataLength] + 1) * CIFAR10[DataAmount]))
    data = np.array(bin_data).reshape(CIFAR10[DataAmount] * (len(CIFAR10[FileName]) - 1), (CIFAR10[DataLength] + 1))
    train_data = data.T[1:].T
    train_label = list(data.T[0])

    # read_test_data
    file_path = os.path.join(DATA_DIR, CIFAR10[FileName][-1])
    with open(file_path, "rb") as f:
        bin_data = list(f.read((CIFAR10[DataLength] + 1) * CIFAR10[TestAmount]))
    data = np.array(bin_data).reshape(CIFAR10[TestAmount], CIFAR10[DataLength] + 1)
    test_data = data.T[1:].T
    test_label = list(data.T[0])
    return train_data, train_label, test_data, test_label


def read_mnist() -> tuple:
    """
    mnistのデータセットを読み込んで配列に格納する
    データは1文字ごとに分けるため2次元配列、ラベルは1次元になる
    :return: train_data, train_label, test_data, test_labelのtuple
    """
    file_path = os.path.join(DATA_DIR, MNIST[FileName][0])
    # read_train_data
    with open(file_path, "rb") as f:
        f.read(16)  # header
        bin_data = f.read(MNIST[DataLength] * MNIST[DataAmount])
    train_data = []
    for i in range(MNIST[DataAmount]):
        train_data.append(list(bin_data[i * MNIST[DataLength]: (i + 1) * MNIST[DataLength]]))
    file_path = os.path.join(DATA_DIR, MNIST[FileName][1])
    # read_train_label
    with open(file_path, "rb") as f:
        f.read(8)  # header
        bin_data = f.read(MNIST[DataAmount])
    train_label = list(bin_data)

    file_path = os.path.join(DATA_DIR, MNIST[FileName][2])
    # read_test_data
    with open(file_path, "rb") as f:
        f.read(16)  # header
        bin_data = f.read(MNIST[DataLength] * MNIST[TestAmount])
    test_data = []
    for i in range(MNIST[TestAmount]):
        test_data.append(list(bin_data[i * MNIST[DataLength]: (i + 1) * MNIST[DataLength]]))

    file_path = os.path.join(DATA_DIR, MNIST[FileName][3])
    # read_test_data
    with open(file_path, "rb") as f:
        f.read(8)  # header
        bin_data = f.read(MNIST[TestAmount])
    test_label = list(bin_data)
    return np.array(train_data), train_label, np.array(test_data), test_label


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


def make_csv(information_dict: dict, dir_name: str):
    """
    学習時の情報をCSV化するメソッド e.g. "MNIST_500-1000_Adam_Sigmoid_11251759.csv"
    :param information_dict: 出力したい全ての情報が入った辞書
    :param dir_name: ディレクトリ名 なければ新たに生成する
    :return: None
    """
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    exec_time = datetime.datetime.now()
    file_title = "{0:%m%d%H%M}.csv".format(exec_time)
    output_dir = os.path.join(OUTPUT_DIR, dir_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_path = os.path.join(output_dir, file_title)
    columns = ["train_accuracy", "train_error", "accuracy", "error", "L1_norm", "L2_norm",
               "node_amount", "epoch_time", "total_time", "weight_active_ratio"]
    with open(file_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for data in zip(*information_dict.values()):
            writer.writerow(data)


if __name__ == "__main__":
    for data_set in [MNIST, CIFAR10]:
        if data_set[DataSet] == "MNIST":
            data_tuple = read_mnist()
        elif data_set[DataSet] == "CIFAR10":
            data_tuple = read_cifar10()
        else:
            print("wrong dataset")
            sys.exit()
        for _ in range(10):
            """
            main(NetWork(hidden_layer=2, in_dim=data_set[DataLength], activation=ReLU, optimizer=ADAM,
                         md1=50, md2=100, out_dim=CLASS_NUM, dynamic=False, propose=False), data_tuple)
            """

            main(NetWork(hidden_layer=2, in_dim=data_set[DataLength], activation=ReLU, optimizer=ADAM,
                         md1=50, md2=100, out_dim=CLASS_NUM, dynamic=False, propose=False), data_tuple)
            main(NetWork(hidden_layer=2, in_dim=data_set[DataLength], activation=ReLU, optimizer=ADAM,
                         md1=50, md2=100, out_dim=CLASS_NUM, dynamic=True, propose=False), data_tuple)
            main(NetWork(hidden_layer=2, in_dim=data_set[DataLength], activation=ReLU, optimizer=ADAM,
                         md1=50, md2=100, out_dim=CLASS_NUM, dynamic=True, propose=True), data_tuple)
