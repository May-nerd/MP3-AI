# VOTING CLASSIFIER (ENSEMBLE)

import Data as dt
import numpy as np
import os
import knearest
import neuralnet
import random_forest

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# GETS THE TRAINING DATA
train_data = dt.train_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = train_data[:,0:10]
y = train_data[:,10]


# INITIALIZING THE CLASSIFIER WITH ITS PARAMETERS
ensemble = VotingClassifier(estimators=[
        ('knn', knearest.knn), ('mlp', neuralnet.clf), ('rf', random_forest.rf)], 
        voting='hard', 
        flatten_transform=None,
        n_jobs=-1)

# FITTING THE CLASSIFIER WITH THE TRAINING DATA        
ensemble = ensemble.fit(X, y)

# GETS THE VALIDATION (OR TEST DEPENDING ON Data.py) DATA
val_data = dt.val_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = val_data[:,0:10]
y_true = val_data[:,10]

# USES THE FITTED CLASSSIFIER TO PREDICT NEW INSTANCES/CASES
y_pred = ensemble.predict(X)

# PRINTS THE ACCURACY COMPARED TO THE TRUE CLASSIFICATION
print("Ensemble: " + str(accuracy_score(y_true, y_pred) * 100))


# TO NOTIFY WHEN THE CODE IS ALREADY DONE (NEEDS Linux spd-say library)
# os.system('spd-say "Done for the ensemble."')
