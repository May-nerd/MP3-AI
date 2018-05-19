import numpy as np

f = open("dataset/train.data")
train_data = np.loadtxt(fname = f)

# CHANGE THIS TO TEST FOR FINAL TESTING
# f = open("dataset/val.data")
f = open("dataset/test.data")
val_data = np.loadtxt(fname = f)
