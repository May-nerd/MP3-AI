import Data as dt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = dt.train_data

X = train_data[:,0:10]
y = train_data[:,10]

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X,y)

val_data = dt.val_data

X = val_data[:,0:10]
y_true = val_data[:,10]
y_pred = knn.predict(X)

print(accuracy_score(y_true, y_pred))
