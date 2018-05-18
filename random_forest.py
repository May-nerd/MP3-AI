import Data as dt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_data = dt.train_data

X = train_data[:,0:10]
y = train_data[:,10]

rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(X, y) 

val_data = dt.val_data

X = val_data[:,0:10]
y_true = val_data[:,10]
y_pred = rf.predict(X)

print(accuracy_score(y_true, y_pred))
