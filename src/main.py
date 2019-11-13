import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error

xtest_raw = pd.read_csv('../airbnb_datasets/xtest.csv')
xtrain_raw = pd.read_csv('../airbnb_datasets/xtrain.csv')
ytest_raw = pd.read_csv('../airbnb_datasets/ytest.csv')
ytrain_raw = pd.read_csv('../airbnb_datasets/ytrain.csv')

xtest = xtest_raw.to_numpy()
xtrain = xtrain_raw.to_numpy()
ytest = ytest_raw.to_numpy()
ytrain = ytrain_raw.to_numpy()

# Linear

# Ridge
# try different alphas and cross validations (k-folds)
clf = Ridge(alpha=1.0)
x = clf.fit(xtrain, ytrain)
y_pred = clf.predict(xtest)
err = mean_squared_error(ytest, y_pred)
print(err)
