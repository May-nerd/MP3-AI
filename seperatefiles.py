from random import shuffle

# fileExtension = "csv"
fileExtension = "data"

fname="dataset/all." + fileExtension
with open(fname) as f:
    content = f.readlines()
content = [x.strip() for x in content] 
shuffle(content)

percentage = (0.70,0.10,0.20)


train = content[0: int(len(content) * percentage[0])]
val = content[int(len(content) * percentage[0]): int(len(content) * (percentage[0]+percentage[1]))]
test = content[int(len(content) * (percentage[0]+percentage[1])):]

print("TRAIN: " + str(len(train)) + " observations.")
print("VAL: " + str(len(val)) + " observations.")
print("TEST: " + str(len(test)) + " observations.")


trainfile = open("dataset/train.data", "w")
for i in train:
    trainfile.write(i + "\n")


valfile = open("dataset/val.data", "w")
for i in val:
    valfile.write(i + "\n")

testfile = open("dataset/test.data", "w")
for i in test:
    testfile.write(i + "\n")

testfile.close()
trainfile.close()
valfile.close()