import os
from consts import *
import time
from classes import *
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math


def main():
    # read_data
    time1 = time.time()
    train_data, train_label, test_data, test_label = read_mnist()
    train_data = np.array(train_data) / 255.0
    test_data = np.array(test_data) / 255.0
    train_data, train_label = transform_data(train_data, train_label)
    # view_data(train_data[0][0], train_label[0][0])
    # view_data(test_data[0], test_label[0])
    model = NetWork(hidden_layer=2)
    model.train_data = train_data
    model.train_labels = train_label
    time2 = time.time()
    print("format_complete, time:{0}".format(time2 - time1))
    model.training()
    time3 = time.time()
    print("train_end, time:{}".format(time3 - time2))
    test_info = model.test(test_data, test_label)
    print("test_end, time:{0}".format(time.time() - time3))
    print("error_average:{0}, recognition_rate:{1}".format(test_info[1], test_info[0] * 100))


def read_mnist() -> tuple:
    """
    mnistのデータセットを読み込んで配列に格納する
    データは1文字ごとに分けるため2次元配列、ラベルは1次元になる
    :return: train_data, train_label, test_data, test_labelのtuple
    """
    file_path = os.path.join(DATA_DIR, MNIST[filename][0])
    # read_train_data
    with open(file_path, "rb") as f:
        f.read(16)  # header
        bin_data = f.read(MNIST[data_length] * MNIST[data_amount])
    train_data = []
    for i in range(MNIST[data_amount]):
        train_data.append(list(bin_data[i * MNIST[data_length]: (i + 1) * MNIST[data_length]]))
    file_path = os.path.join(DATA_DIR, MNIST[filename][1])
    # read_train_label
    with open(file_path, "rb") as f:
        f.read(8)  # header
        bin_data = f.read(MNIST[data_amount])
    train_label = list(bin_data)

    file_path = os.path.join(DATA_DIR, MNIST[filename][2])
    # read_test_data
    with open(file_path, "rb") as f:
        f.read(16)  # header
        bin_data = f.read(MNIST[data_length] * MNIST[test_amount])
    test_data = []
    for i in range(MNIST[test_amount]):
        test_data.append(list(bin_data[i * MNIST[data_length]: (i + 1) * MNIST[data_length]]))

    file_path = os.path.join(DATA_DIR, MNIST[filename][3])
    # read_test_data
    with open(file_path, "rb") as f:
        f.read(8)  # header
        bin_data = f.read(MNIST[test_amount])
    test_label = list(bin_data)
    return train_data, train_label, test_data, test_label


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


def view_data(data: np.array, data_label: int):
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
    plt.xlabel(data_label)
    plt.show()
    img.show()


if __name__ == "__main__":
    main()
