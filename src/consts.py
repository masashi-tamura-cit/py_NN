import os
# Network params
BATCH_SIZE = 4
DIM = 784
MD1 = 50
MD2 = 100
CLASS_NUM = 10

# Hyper params
ETA = 0.01

# Dataset
dataset = 0
filename = 1
label = 2
data_length = 3
data_amount = 4
test_amount = 5
DATA_DIR = os.path.join(os.getcwd(), "data")

MNIST = {
    dataset: "MNIST",
    filename: ["train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"],
    label: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    data_length: 784, data_amount: 60000, test_amount: 10000
}

CIFAR10 = {
    dataset: "CIFAR10",
    filename: ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
               "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"],
    label: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    data_length: 3072, data_amount: 10000, test_amount: 10000
}

CIFAR100 = {dataset: "CIFAR100", filename: ["train.bin", "test.bin"]}
