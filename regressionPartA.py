import importlib_resources
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
from utils import monthToNum, dayToNum
import sklearn.linear_model as lm
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    loglog,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)
from scipy.io import loadmat
from sklearn import model_selection

from dtuimldmtools import rlr_validate


#Read CSV file.
df = pd.read_csv("forestfires.csv")

#Log transform the area
y_t = df["area"]

y = (y_t+1).apply(math.log)

#Convert month column to integer
month_column = df["month"]
month_column_int = month_column.apply(monthToNum)
df["month"] = month_column_int

#Convert day column to 0 if work day or 1 if weekend
day_column = df["day"]
day_column_int = day_column.apply(dayToNum)
df["day"] = day_column_int


X = df.loc[:, df.columns != "month"].values

#One-out-of-K encoding for month
month = np.array(df["month"].values, dtype=int).T
K = month.max()
month_encoding = np.zeros((month.size, K))
month_encoding[np.arange(month.size), month-1] = 1
X = np.concatenate((X[:,:-1], month_encoding), axis=1)


#Standardize the data
X = (X - np.ones((len(y), 1)) * X.mean(0)) / X.std(0)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
#attributeNames = ["Offset"] + attributeNames
M = M + 1

## Crossvalidation
# Amount of folds
K = 10

# Values of lambda
lambdas = np.power(10.0, [2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5])

(
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X, y, lambdas, K)

# Plot of Lambda vs. generalization error
loglog(lambdas, test_err_vs_lambda.T)
xlabel("Regularization Factor (Lambda)")
ylabel("Estimated Generalization Error")
plt.show()

# Plot of weights vs. lambda
semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], ".-")
xlabel("Regularization factor")
ylabel("Mean Coefficient Values")
grid()
plt.show()

#Printing the optimal lambda
print(f"The optimal lambda is: {opt_lambda}")

# Calculating the weights based on all data
lambdaI = opt_lambda * np.eye(M)
Xty = X.T @ y
XtX = X.T @ X
weights = np.linalg.solve(XtX + lambdaI, Xty)
print(weights)