import Data as dt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_data = dt.train_data

X = train_data[:,0:10]
y = train_data[:,10]

svm = SVC()
svm.fit(X, y) 

val_data = dt.val_data

X = val_data[:,0:10]
y_true = val_data[:,10]
y_pred = svm.predict(X)

print(accuracy_score(y_true, y_pred))
