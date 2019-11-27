import csv
import os
from consts import *
import time
from classes import *
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
import random
import sys
import datetime


def main(data_set):
    exec_time = datetime.datetime.now()
    models = []
    # read_data
    time1 = time.time()
    if data_set[DataSet] == "MNIST":
        train_data, train_label, test_data, test_label = read_mnist()
    elif data_set[DataSet] == "CIFAR10":
        train_data, train_label, test_data, test_label = read_cifar10()
    else:
        print("wrong dataset")
        sys.exit()
    models.append(NetWork(hidden_layer=2, input_dim=data_set[DataLength], activation=ReLU, optimizer=ADAM))
    # models.append(NetWork(hidden_layer=2, input_dim=data_set[DataLength], activation=ReLU, optimizer=SGD))
    # models.append(NetWork(hidden_layer=2, input_dim=data_set[DataLength], activation=ReLU, optimizer=MOMENTUM_SGD))
    # models.append(NetWork(hidden_layer=2, input_dim=data_set[DataLength], activation=SIGMOID, optimizer=ADAM))
    # models.append(NetWork(hidden_layer=2, input_dim=data_set[DataLength], activation=SIGMOID, optimizer=SGD))
    # models.append(NetWork(hidden_layer=2, input_dim=data_set[DataLength], activation=SIGMOID, optimizer=MOMENTUM_SGD))
    time2 = time.time()
    # print("format_complete time:{0}".format(time2 - time1))
    for model in models:
        time2 = time.time()
        c = 0
        train_accuracy = []
        train_error = []
        accuracy = []
        err = []
        l1_norm = []
        l2_norm = []
        node_amount = []
        start = 0
        is_proved = True
        while is_proved:
            while not early_stopping(err, start):
                time3 = time.time()
                # print("epoch{0} train_start".format(c))
                data, label = shuffle_data(train_data[:SAMPLE_SIZE], train_label[:SAMPLE_SIZE])
                # data, label = shuffle_data(train_data, train_label)
                data, label = transform_data(data, label)
                train_info = model.training(data, label)
                train_accuracy.append(train_info[0])
                train_error.append(train_info[1])
                time4 = time.time()
                print("train_end, time:{}".format(time4 - time3))
                test_info = model.test(test_data[:VALIDATION_DATA], test_label[:VALIDATION_DATA])
                time5 = time.time()
                print("test_end, time:{0}".format(time5 - time4))
                print("error_average:{0}, accuracy:{1}%".format(test_info[1], int(test_info[0] * 100)))
                print("l1_norm:{0}, l2_norm:{1}, node_amount{2}".format(test_info[2], test_info[3], test_info[4]))
                accuracy.append(test_info[0])
                err.append(test_info[1])
                l1_norm.append(test_info[2])
                l2_norm.append(test_info[3])
                node_amount.append(test_info[4])
                c += 1
                if not (c - start) % 5:
                    model.propose_method(err[-5], err[-1])
                # plot_fig(accuracy, err, l1_norm, l2_norm, node_amount, c, model)
            # print("early_stopping, epochs: {0}".format(c-start))
            if model.is_proved(accuracy):
                model.add_layer()
                # print("add_layer")
                start = c
            else:
                is_proved = False
        # additional leaning
        model.rollback_layer()
        # print("decision layer num, start additional learning")
        for i in range(10):
            data, label = shuffle_data(train_data[:SAMPLE_SIZE], train_label[:SAMPLE_SIZE])
            data, label = transform_data(data, label)
            model.training(data, label, 2)
            test_info = model.test(test_data[:VALIDATION_DATA], test_label[:VALIDATION_DATA])
            accuracy.append(test_info[0])
            err.append(test_info[1])
            l1_norm.append(test_info[2])
            l2_norm.append(test_info[3])
            node_amount.append(test_info[4])
            c += 1
        for i in range(10):
            data, label = shuffle_data(train_data[:SAMPLE_SIZE], train_label[:SAMPLE_SIZE])
            data, label = transform_data(data, label)
            model.training(data, label, 1)
            test_info = model.test(test_data[:VALIDATION_DATA], test_label[:VALIDATION_DATA])
            accuracy.append(test_info[0])
            err.append(test_info[1])
            l1_norm.append(test_info[2])
            l2_norm.append(test_info[3])
            node_amount.append(test_info[4])
            c += 1

        test_info = model.test(test_data[:VALIDATION_DATA], test_label[:VALIDATION_DATA])
        time5 = time.time()
        # print("\ntotal_train_time:{0}".format(time5 - time2))
        # print("total_epoch:{0}".format(c))
        print("latest accuracy:{0}%".format(int(test_info[0]*100)))
        # print("latest error:{0}".format(test_info[1]))

        title = plot_fig(accuracy, err, l1_norm, l2_norm, node_amount, c, model)
        make_csv(title, exec_time, train_accuracy, train_error, accuracy, err, l1_norm, node_amount)
    print("total {0}min".format(int((time5 - time1)/60)))


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


def view_mnist(data: np.array, label: int) -> None:
    """
    mnistのデータを画像出力するメソッド
    :param data:MNSITのバイナリデータ配列
    :param label:dataのラベル
    :return:
    """
    x = []
    y = []
    size = data.size
    dim1 = int(np.sqrt(size))
    img = Image.new(mode='RGB', size=(dim1, dim1))
    for i in range(size):
        cr = int(data[i]*255)
        if data[i] != 0:
            y.append(dim1 - math.floor(i/dim1))
            x.append(i % dim1)
            img.putpixel((i % dim1, math.floor(i/dim1)), (cr, cr, cr))
    plt.scatter(x, y)
    plt.xlabel(label)
    plt.show()
    img.show()


def view_cifar(data: np.array, label: int) -> None:
    """
    cifarのデータを画像出力するメソッド
    :param data: CIFARのバイナリデータ配列
    :param label: dataのラベル
    :return:
    """
    color_data = data.reshape((3, 1024))
    img = Image.new(mode='RGB', size=(32, 32))
    for i in range(1024):
        r = color_data[0][i]
        g = color_data[1][i]
        b = color_data[2][i]
        img.putpixel((i % 32, math.floor(i / 32)), (r, g, b))
    print(CIFAR10[Label][label])
    img.show()


def early_stopping(err: list, start) -> bool:
    if len(err) == start:
        return False
    arg_min = np.argmin(np.array(err[start:]))
    if (len(err[start:]) - arg_min) < EARLY_STOPPING_EPOCH:
        return False
    return True


def plot_fig(accuracy: list, err: list, l1_norm: list, l2_norm: list, total_amount: list, c: int, model: NetWork):

    name = "MNIST" if model.input_dim == 784 else "CIFAR10"
    activation = "Sigmoid" if model.activation == SIGMOID else "ReLU"

    if isinstance(model.optimizer, Adam):
        optimizer = "Adam"
    elif isinstance(model.optimizer, MomentumSgd):
        optimizer = "MomentumSGD"
    else:
        optimizer = "SGD"
    title = "{0}, {1}-{2}, {3}, {4}".format(name, MD1, MD2, optimizer, activation)  # e.g. MNIST, 50-100, Adam, Sigmoid
    fig = plt.figure()
    fig.suptitle(title)
    performance_fig = fig.add_subplot(2, 1, 1)
    status_fig = fig.add_subplot(2, 1, 2)
    x = list(range(0, c, 1))

    performance_fig.plot(x, accuracy, color="g", label="accuracy")
    performance_fig.set_ylim(0, 1)
    error_fig = performance_fig.twinx()
    error_fig.plot(x, err, color="b", linestyle="dotted", label="error")
    handle1, label1 = performance_fig.get_legend_handles_labels()
    handle2, label2 = error_fig.get_legend_handles_labels()
    performance_fig.legend(handle1 + handle2, label1 + label2)
    # performance_fig.set_title("performance")

    status_fig.plot(x, l1_norm, color="g", label="L1_norm")
    # status_fig.plot(x, l2_norm, color="b", label="L2_norm")
    node_amount = status_fig.twinx()
    node_amount.plot(x, total_amount, color="r", linestyle="dotted", label="node_amount")
    handle1, label1 = status_fig.get_legend_handles_labels()
    handle2, label2 = node_amount.get_legend_handles_labels()
    status_fig.legend(handle1 + handle2, label1 + label2)
    # status_fig.set_title("status")
    # plt.subplots_adjust(top=0.85)
    plt.show()
    return title


def make_csv(title, exec_time, train_accuracy, train_error, accuracy, error, l1_norm, node_amount):
    """
    学習時の情報をCSV化するメソッド e.g. "MNIST_500-1000_Adam_Sigmoid_11251759.csv"
    :param title: ファイル名 str
    :param exec_time: スクリプト実行時刻 datetime
    :param train_accuracy: 訓練時の正答率
    :param train_error: 訓練時の誤差関数地
    :param accuracy: 精度 list
    :param error: 誤差関数値 list
    :param l1_norm: 重みの絶対値和 list
    :param node_amount: ノード数 list
    :return: None
    """
    columns = ["train_accuracy", "train_error", "accuracy", "error", "L1_norm", "node_amount"]
    file_title = "{0}_{1:%m%d%H%M}.csv".format(title, exec_time).replace(", ", "_")
    file_path = os.path.join(DATA_DIR, file_title)
    with open(file_path, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for data in zip(train_accuracy, train_error, accuracy, error, l1_norm, node_amount):
            writer.writerow(data)


if __name__ == "__main__":
    # print("IMPORTANT!")
    for _ in range(1):
        main(MNIST)
        # main(CIFAR10)
