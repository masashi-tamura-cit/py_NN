import os
from consts import *
import time
from classes import *
import numpy as np


def main():
    # read_data
    time1 = time.time()
    train_data, train_label, test_data, test_label = read_mnist()
    train_data = np.array(train_data) / 255.0
    test_data = np.array(test_data) / 255.0
    model = NetWork(hidden_layer=2)
    time2 = time.time()
    print("format_complete, time:{0}".format(time2 - time1))
    for i in range(0, MNIST[data_amount], BATCH_SIZE):
        model.train_data_input(train_data[i: i + BATCH_SIZE], train_label[i:i + BATCH_SIZE])
        model.training()
    time3 = time.time()
    print("train_end, time:{}".format(time3 - time2))
    test_info = model.test(test_data, test_label)
    print("test_end, time:{0}, recognition_rate:{1}".format(time.time() - time3, test_info * 100))


def read_mnist() -> tuple:
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


if __name__ == "__main__":
    main()
