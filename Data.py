import numpy as np

# EXTRACTS TRAINING DATA
f = open("dataset/train.data")
train_data = np.loadtxt(fname = f)


# UNCOMMENT NEXT LINE FOR VALIDATION
# f = open("dataset/val.data")

# UNCOMMENT NEXT LINE FOR FINAL TESTING
f = open("dataset/test.data")
val_data = np.loadtxt(fname = f)
