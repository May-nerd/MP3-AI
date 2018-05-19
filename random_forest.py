# Random Forest Classifier

import Data as dt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# THESE CODE IS FOR THE EXPERIMENTS DONE (LOOPING OVER DIFFERENT CONFIGURATIONS)
# for m_depth in [None, 1, 5, 10]:
#     for n_esti in [10,20,30]: 

# GETS THE TRAINING DATA
train_data = dt.train_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = train_data[:,0:10]
y = train_data[:,10]

# INITIALIZING THE CLASSIFIER WITH ITS PARAMETERS
rf = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=None, n_estimators=81)

# FITTING THE CLASSIFIER WITH THE TRAINING DATA
rf.fit(X, y) 


# GETS THE VALIDATION (OR TEST DEPENDING ON Data.py) DATA
val_data = dt.val_data

# SEPARATES THE CLASS VARIABLE (y) AND THE FEATURES (X)
X = val_data[:,0:10]
y_true = val_data[:,10]

# USES THE FITTED CLASSSIFIER TO PREDICT NEW INSTANCES/CASES
y_pred = rf.predict(X)

# PRINTS THE ACCURACY COMPARED TO THE TRUE CLASSIFICATION
print("RF: " + str(accuracy_score(y_true, y_pred) * 100))
