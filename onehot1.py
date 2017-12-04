
import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([[0], [1], [0],[2]])
print "enc.n_values_ is:",enc.n_values_
print "enc.feature_indices_ is:",enc.feature_indices_
print enc.transform([[0]]).toarray()
print enc.transform([[2]]).toarray()
print enc.transform([[1],[1],[1],[0],[2]]).toarray()
print(type(enc.transform([[1],[1],[1],[0],[2]]).toarray()))
print((enc.transform([[1],[1],[1],[0],[2]]).toarray()).shape)

