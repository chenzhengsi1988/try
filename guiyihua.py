import numpy as np
from sklearn.preprocessing import MinMaxScaler
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
min_max_scaler = MinMaxScaler()
print min_max_scaler
#MinMaxScaler(copy=True, feature_range=(0, 1))
X_train_minmax = min_max_scaler.fit_transform(X_train)
print X_train_minmax
#[[ 0.5         0.    