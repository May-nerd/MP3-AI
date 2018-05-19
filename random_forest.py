import Data as dt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# for m_depth in [None, 1, 5, 10]:
#     for n_esti in [10,20,30]: 
train_data = dt.train_data

X = train_data[:,0:10]
y = train_data[:,10]

rf = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=None, n_estimators=81)
rf.fit(X, y) 

val_data = dt.val_data

X = val_data[:,0:10]
y_true = val_data[:,10]
y_pred = rf.predict(X)
# print()
# print("Max depth: " + str(m_depth))
# print("N-estimators: " + str(n_esti))
print("RF: " + str(accuracy_score(y_true, y_pred) * 100))
