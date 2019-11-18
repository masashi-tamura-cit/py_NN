import os
# Network params
BATCH_SIZE = 32
MD1 = 50
MD2 = 10
CLASS_NUM = 10

# Hyper params
ETA = 0.01
EarlyStopping = 50

# Dataset
DataSet = 0
FileName = 1
Label = 2
DataLength = 3
DataAmount = 4
TestAmount = 5
# DATA_DIR = os.path.join(os.getcwd(), "data")
DATA_DIR = 'C:\\Users\\tamtam\\PycharmProjects\\untitled\\src\\data'
SAMPLE_SIZE = 32000

MNIST = {
    DataSet: "MNIST",
    FileName: ["train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte"],
    Label: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    DataLength: 784, DataAmount: 60000, TestAmount: 10000
}

CIFAR10 = {
    DataSet: "CIFAR10",
    FileName: ["data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
               "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"],
    Label: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    DataLength: 3072, DataAmount: 10000, TestAmount: 10000
}

CIFAR100 = {DataSet: "CIFAR100", FileName: ["train.bin", "test.bin"]}

ReLU = 0
SIGMOID = 1
ACTIVATE = ReLU
BATCH_NORM = True

SGD = 0
MomentumSGD = 1
Adam = 2
OPTIMIZER = SGD
