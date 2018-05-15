import Data as dt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

train_data = dt.train_data


X = train_data[:,1:]
y = train_data[:,0]

print(X)
print(X.shape)


print(y)
print(y.shape)


knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X,y)


example =  np.array([227,8,2,-1,0,-1,0,0,0,0,0,1,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
example = example.reshape(1, -1)
print(knn.predict(example))

