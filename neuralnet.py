import Data as dt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
import os

solvers = ['lbfgs', 'sgd', 'adam']
act_functions = ['identity', 'logistic', 'tanh', 'relu']


# for i in solvers:
#     for j in act_functions:

train_data = dt.train_data

X = train_data[:,:10]
y = train_data[:,10]

# print(X.shape)
# print(y.shape)

clf = MLPClassifier(solver='lbfgs', activation='logistic', hidden_layer_sizes=(600,), random_state=0, early_stopping=True, max_iter=1000, learning_rate='adaptive')
clf.fit(X, y)


val_data = dt.val_data

X = val_data[:,:10]
y_true = val_data[:,10]
y_pred = clf.predict(X)
print("MLP: " + str(accuracy_score(y_true, y_pred) * 100))


os.system('spd-say "training is finished"')
