import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from yellowbrick.regressor import AlphaSelection
from yellowbrick.regressor import PredictionError
import shap

xtest_raw = pd.read_csv('../airbnb_datasets/xtest.csv')
xtrain_raw = pd.read_csv('../airbnb_datasets/xtrain.csv')
ytest_raw = pd.read_csv('../airbnb_datasets/ytest.csv')
ytrain_raw = pd.read_csv('../airbnb_datasets/ytrain.csv')

xtest = np.array(xtest_raw)
xtrain = np.array(xtrain_raw)
ytest = np.array(ytest_raw)
ytrain = np.array(ytrain_raw)

# Baseline model
ridge_reg = Ridge(alpha=1)
x = ridge_reg.fit(xtrain, ytrain)
y_pred = ridge_reg.predict(xtest)
err = mean_squared_error(ytest, y_pred)
print("MSE for Baseline model: ", err)

# # Ridge Regression Dataframe
# ridge_df = pd.DataFrame({'variable': list(xtrain_raw.columns), 'estimate': ridge_reg.coef_[0]})
# ridge_train_pred = []
# ridge_test_pred = []
# mse_list = list()
#
# # Manually Iterate through the lambdas to find the optimal
# alphas = np.arange(0.01, 5, 0.01)
# for alpha in alphas:
#     # training
#     ridge_reg = Ridge(alpha=alpha)
#     ridge_reg.fit(xtrain, ytrain)
#     var_name = 'estimate' + str(alpha)
#     ridge_df[var_name] = ridge_reg.coef_[0]
#     # prediction
#     y_pred = ridge_reg.predict(xtest)
#     ridge_test_pred.append(y_pred)
#     ridge_train_pred.append(ridge_reg.predict(xtrain))
#     mse_list.append(mean_squared_error(ytest, y_pred))
#
# min_mse = min(mse_list)
# min_mse_index = mse_list.index(min_mse)
# optimal_alpha = alphas[min_mse_index]
# print("Optimal Alpha: ", optimal_alpha)
# print("Minimum MSE: ", min_mse)
#
# plt.scatter(alphas, mse_list)
# plt.ylim((min(mse_list) - 0.0001, max(mse_list) + 0.0001))
# plt.show()

# Yellowbrick Regressor - Predict optimal alpha
ytrain = np.reshape(ytrain, (ytrain.shape[0]))
alphas = np.logspace(-10, 1, 200)
visualizer = AlphaSelection(RidgeCV(alphas=alphas))
visualizer.fit(xtrain, ytrain)
visualizer.show()

# Optimal model
optimal_alpha = 4.103
ridge_reg = RidgeCV(alphas=np.array([optimal_alpha]))
x = ridge_reg.fit(xtrain, ytrain)

# Yellowbrick Regressor - Plot error
visualizer = PredictionError(ridge_reg)
visualizer.fit(xtrain, ytrain)
visualizer.score(xtest, ytest)
visualizer.show()

# SHAP Values
explainer = shap.LinearExplainer(ridge_reg, xtrain)
shap_values = explainer.shap_values(xtest)
shap.summary_plot(shap_values, xtest, plot_type='bar')
feature_indices = [227, 5, 0, 228, 133, 101, 220, 208, 2, 70, 1, 40, 207, 229, 215, 79, 4, 125, 100, 98]
for i in feature_indices:
    print("feature ", i, ": ", xtrain_raw.columns[i])

# # Plot betas by lambda
# fig, ax = plt.subplots(figsize=(10, 5))
# # ax.plot(ridge_df.RM, 'r', ridge_df.ZN, 'g', ridge_df.RAD, 'b', ridge_df.CRIM, 'c', ridge_df.TAX, 'y')
# ax.plot(ridge_df.minimum_nights, 'r')
# ax.axhline(y=0, color='black', linestyle='--')
# ax.set_xlabel("Lambda")
# ax.set_ylabel("Beta Estimate")
# ax.set_title("Ridge Regression Trace", fontsize=16)
# # ax.legend(labels=['Room','Residential Zone','Highway Access','Crime Rate','Tax'])
# ax.grid(True)
# plt.show()

# # MSE of Ridge and OLS
# ridge_mse_test = [mean_squared_error(ytest, p) for p in ridge_test_pred]
# ols_mse = mean_squared_error(ytest, ols_pred)
#
# # plot mse
# plt.plot(ridge_mse_test[:25], 'ro')
# plt.axhline(y=ols_mse, color='g', linestyle='--')
# plt.title("Ridge Test Set MSE", fontsize=16)
# plt.xlabel("Model Simplicity$\longrightarrow$")
# plt.ylabel("MSE")
# plt.show()
