import Data as dt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = dt.train_data

# X = train_data[:,0:10]
# y = train_data[:,10]


# for i in [1,2,3,4,5,10,50,100,200,500,1000]:
X = train_data[:,0:10]
y = train_data[:,10]

knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1, weights='distance')
knn.fit(X,y)
val_data = dt.val_data
X = val_data[:,0:10]
y_true = val_data[:,10]
y_pred = knn.predict(X)
# print()
print("KNN: " + str(accuracy_score(y_true, y_pred) * 100))
# print(accuracy_score(y_true, y_pred))




	
# n_neighbors : int, optional (default = 5)

# weights 

# njobs