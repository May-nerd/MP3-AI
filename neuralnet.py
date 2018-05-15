import Data as dt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
import os


train_data = dt.train_data

X = train_data[:,:10]
y = train_data[:,10]

# print(X.shape)
# print(y.shape)

clf = MLPClassifier(solver='sgd',hidden_layer_sizes=(300,), random_state=1, verbose=True, early_stopping=True, max_iter=1000, learning_rate='adaptive')
clf.fit(X, y)

# example =  np.array([[227,8,2,-1,0,-1,0,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[227,8,2,-1,0,-1,0,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]])
# exa scaler.transform(example)
# print(clf.predict(example))




test_data = dt.test_data

X = test_data[:,:10]
y_true = test_data[:,10]
y_pred = clf.predict(X)


print(accuracy_score(y_true, y_pred))

os.system('spd-say "training is finished"')
