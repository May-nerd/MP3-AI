import numpy as np

# DATA
f = open("dataset/train.data")
train_data = np.loadtxt(fname = f)

f = open("dataset/val.data")
val_data = np.loadtxt(fname = f)

f = open("dataset/test.data")
test_data = np.loadtxt(fname = f)
