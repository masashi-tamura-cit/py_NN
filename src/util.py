import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math
from consts import *
from classes import *


def plot_fig(accuracy: list, err: list, l1_norm: list, l2_norm: list, total_amount: list, c: int, model: NetWork):

    name = "MNIST" if model.layer_dims[0] == 784 else "CIFAR10"
    activation = "Sigmoid" if model.activation == SIGMOID else "ReLU"

    if isinstance(model.optimizer, Adam):
        optimizer = "Adam"
    elif isinstance(model.optimizer, MomentumSgd):
        optimizer = "MomentumSGD"
    else:
        optimizer = "SGD"
    title = "{0}, {1}, {2}".format(name, optimizer, activation)  # e.g. MNIST, 50-100, Adam, Sigmoid
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
    # plt.show()
    return title


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
