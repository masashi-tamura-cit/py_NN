import os
from consts import *
import time
from classes import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines
from PIL import Image
import math
import random


def main():
    # read_data
    time1 = time.time()
    # train_data, train_label, test_data, test_label = read_cifar10()
    train_data, train_label, test_data, test_label = read_mnist()
    # model = NetWork(hidden_layer=2, input_dim=3072)  # CIFAR
    model = NetWork(hidden_layer=2, input_dim=784)  # MNIST
    time2 = time.time()
    print("format_complete time:{0}".format(time2 - time1))
    accuracy = []
    c = 0
    while True:
        time3 = time.time()
        print("epoc{0} train_start".format(c))
        data, label = shuffle_data(train_data[:SAMPLE_SIZE], train_label[:SAMPLE_SIZE])
        data, label = transform_data(data, label)
        model.training(data, label)
        time4 = time.time()
        print("train_end, time:{}".format(time4 - time3))
        test_info = model.test(test_data[:1024], test_label[:1024])
        time5 = time.time()
        print("test_end, time:{0}".format(time5 - time4))
        print("error_average:{0}, accuracy:{1}".format(test_info[1], test_info[0] * 100))
        accuracy.append(test_info[0])
        c += 1
        if early_stopping(c):
            break
    print("\ntotal_train_time:{0}".format(time5 - time2))
    print("total_epoc:{0}".format(c))
    print("latest accuracy:{0}".format(test_info[0]))
    print("latest error:{0}".format(test_info[1]))
    plt.plot(list(range(0, c, 1)), accuracy)
    plt.show()


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


def transform_data(train_data: np.array, train_label: list) -> tuple:
    """
    データを3次元配列に整形する np.array([[[],[],[]],[[],[],[]]]) 的なイメージ
    :param train_data: 学習データの2次元配列
    :param train_label: 学習データのラベル
    :return: 整形後のデータとラベルのタプル
    """
    dim = int(train_data[0].size)
    batch_num = int(train_data.size/(BATCH_SIZE * dim))
    return_data = train_data.reshape((batch_num, BATCH_SIZE, dim))
    return_label = np.array(train_label).reshape((batch_num, BATCH_SIZE)).tolist()
    return return_data, return_label


def shuffle_data(data: np.array, label: list) -> tuple:
    """
    受け取ったデータ配列とラベル配列の順番をランダムに入れ替える
    :param data: データの2次元配列
    :param label: ラベルの配列
    :return: ランダムに入れ替えたtrain_dataとそれに対応したtrain_label
    """
    shape = data.shape
    array = np.reshape(np.append(data.T, label), (shape[1] + 1, shape[0])).T.tolist()
    array = np.array(random.sample(array, SAMPLE_SIZE))
    return_data = array.T[:-1].T
    return_label = list(array.T[-1])

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


def early_stopping(counter: int) -> bool:
    if counter < EarlyStopping:
        return False
    return True


if __name__ == "__main__":
    main()
