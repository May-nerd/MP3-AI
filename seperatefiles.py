from random import shuffle

# fileExtension = "csv"
fileExtension = "data"

fname="dataset/all." + fileExtension
with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content] 
shuffle(content)

train_percent = 0.9

train = content[0: int(len(content) * train_percent)]
test = content[int(len(content) * (train_percent)):]

print("TRAIN: " + str(len(train)) + " observations.")
print("TEST: " + str(len(test)) + " observations.")


trainfile = open("dataset/train."+fileExtension, "w")
for i in train:
    trainfile.write(i + "\n")

testfile = open("dataset/test."+fileExtension, "w")
for i in test:
    testfile.write(i + "\n")

testfile.close()
trainfile.close()