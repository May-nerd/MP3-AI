import numpy as np



f = open("dataset/train.csv")
train_data = np.loadtxt(fname = f, delimiter = ',')

f = open("dataset/val.csv")
val_data = np.loadtxt(fname = f, delimiter = ',')

f = open("dataset/test.csv")
test_data = np.loadtxt(fname = f, delimiter = ',')