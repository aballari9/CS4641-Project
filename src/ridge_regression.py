# Ridge Regression
from sklearn.linear_model import Ridge
import numpy as np

clf = Ridge(alpha=1.0)
clf.fit(X, y)
