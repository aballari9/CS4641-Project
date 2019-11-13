import pandas as pd

xtest_raw = pd.read_csv('../airbnb_datasets/xtest.csv')
xtrain_raw = pd.read_csv('../airbnb_datasets/xtrain.csv')
ytest_raw = pd.read_csv('../airbnb_datasets/ytest.csv')
ytrain_raw = pd.read_csv('../airbnb_datasets/ytrain.csv')

xtest_raw = xtest_raw.to_numpy()
xtrain_raw = xtrain_raw.to_numpy()
ytest_raw = ytest_raw.to_numpy()
ytrain_raw = ytrain_raw.to_numpy()
