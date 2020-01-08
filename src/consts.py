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
SGD = 0
MOMENTUM_SGD = 1
ADAM = 2

# settings
DATASET = MNIST
BATCH_SIZE = 32
SAMPLE_SIZE = 6400
VALIDATION_DATA = 100
EARLY_STOPPING_EPOCH = 30
# optimizer params
ETA = 0.001
ALPHA = 0.01
BETA1 = 0.9
BETA2 = 0.999
EPS = pow(10, -8)
LAMBDA = 0  # for norm
OUTPUT_BASE_DIR = os.path.join(DATA_DIR, "output")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "output_lm{0}".format(LAMBDA).replace(".", ""))
