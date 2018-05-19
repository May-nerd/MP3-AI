import Data as dt
import numpy as np
import os
import knearest
import neuralnet
import random_forest

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

train_data = dt.train_data

X = train_data[:,0:10]
y = train_data[:,10]



ensemble = VotingClassifier(estimators=[
        ('knn', knearest.knn), ('mlp', neuralnet.clf), ('rf', random_forest.rf)], 
        voting='hard', 
        flatten_transform=None,
        n_jobs=-1)
ensemble = ensemble.fit(X, y)

val_data = dt.val_data

X = val_data[:,0:10]
y_true = val_data[:,10]
y_pred = ensemble.predict(X)
print("Ensemble: " + str(accuracy_score(y_true, y_pred) * 100))





os.system('spd-say "Done for the ensemble. Hahahahahaha"')
