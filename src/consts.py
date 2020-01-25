import os
# Network params
CLASS_NUM = 10

# Dataset
DataSet = 0
FileName = 1
Label = 2
DataLength = 3
DataAmount = 4
TestAmount = 5
DATA_DIR = os.path.join(os.getcwd(), "data")

MNIST = {
    DataSet: "MNIST",
    FileName: ["train.csv", "test.csv"],
    Label: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    DataLength: 784, DataAmount: 60000, TestAmount: 10000
}

ReLU = 0
SIGMOID = 1
SGD = 0
MOMENTUM_SGD = 1
ADAM = 2

# settings
DATASET = MNIST
BATCH_SIZE = 32
SAMPLE_SIZE = 40000
VALIDATION_DATA = 1000
EARLY_STOPPING_EPOCH = 30
# optimizer params
ETA = 0.01
MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPS = pow(10, -8)

LAMBDA = 0.01  # for norm
