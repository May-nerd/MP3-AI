import knearest
import numpy as np
import Data as dt
from sklearn.metrics import accuracy_score
import os


knn = knearest.knn


val_data = dt.test_data

X = val_data[:,1:]
y_true = val_data[:,0]
y_pred = knn.predict(X)


print(accuracy_score(y_true, y_pred))

os.system('spd-say "your program has finished"')
