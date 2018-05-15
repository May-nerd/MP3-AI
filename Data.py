import numpy as np

# # CSV
# f = open("dataset/train.csv")
# train_data = np.loadtxt(fname = f, delimiter = '')


# f = open("dataset/test.csv")
# test_data = np.loadtxt(fname = f, delimiter = ',')




# DATA
f = open("dataset/train.data")
train_data = np.loadtxt(fname = f)

f = open("dataset/test.data")
test_data = np.loadtxt(fname = f)
print(test_data)