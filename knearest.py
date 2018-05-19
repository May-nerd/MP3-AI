# KNEAREST Classifier

import Data as dt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# THESE CODE IS FOR THE EXPERIMENTS DONE (LOOPING OVER DIFFERENT CONFIGURATIONS)
# for i in [1,2,3,4,5,10,50,100,200,500,1000]:

# GETS THE TRAINING DATA
train_data = dt.train_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = train_data[:,0:10]
y = train_data[:,10]

# INITIALIZING THE CLASSIFIER WITH ITS PARAMETERS
knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1, weights='distance')

# FITTING THE CLASSIFIER WITH THE TRAINING DATA
knn.fit(X,y)

# GETS THE VALIDATION (OR TEST DEPENDING ON Data.py) DATA
val_data = dt.val_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = val_data[:,0:10]
y_true = val_data[:,10]

# USES THE FITTED CLASSSIFIER TO PREDICT NEW INSTANCES/CASES
y_pred = knn.predict(X)

# PRINTS THE ACCURACY COMPARED TO THE TRUE CLASSIFICATION
print("KNN: " + str(accuracy_score(y_true, y_pred) * 100))
