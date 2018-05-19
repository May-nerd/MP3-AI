# MLP Neural Network Classifier

import Data as dt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
import os

# THESE CODE IS FOR THE EXPERIMENTS DONE (LOOPING OVER DIFFERENT CONFIGURATIONS)
solvers = ['lbfgs', 'sgd', 'adam']
act_functions = ['identity', 'logistic', 'tanh', 'relu']
# for i in solvers:
#     for j in act_functions:

# GETS THE TRAINING DATA
train_data = dt.train_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = train_data[:,:10]
y = train_data[:,10]

# INITIALIZING THE CLASSIFIER WITH ITS PARAMETERS
clf = MLPClassifier(solver='lbfgs', activation='logistic', hidden_layer_sizes=(600,), random_state=0, early_stopping=True, max_iter=1000, learning_rate='adaptive')

# FITTING THE CLASSIFIER WITH THE TRAINING DATA
clf.fit(X, y)

# GETS THE VALIDATION (OR TEST DEPENDING ON Data.py) DATA
val_data = dt.val_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = val_data[:,:10]
y_true = val_data[:,10]

# USES THE FITTED CLASSSIFIER TO PREDICT NEW INSTANCES/CASES
y_pred = clf.predict(X)

# PRINTS THE ACCURACY COMPARED TO THE TRUE CLASSIFICATION
print("MLP: " + str(accuracy_score(y_true, y_pred) * 100))

# TO NOTIFY WHEN THE CODE IS ALREADY DONE (NEEDS Linux spd-say library)
# os.system('spd-say "training is finished"')
