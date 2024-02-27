# exercise 2.1.1
import math
import importlib_resources
import numpy as np
import pandas as pd
from utils import monthToNum, dayToNum
from scipy.linalg import svd
from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel

# Load xls sheet with data
df = pd.read_csv("forestfires.csv")

#Log transform the area
#TODO: Show that this is necessary and it corrects the normal distribution.
y = (df["area"]+1).apply(math.log)


#Convert month column to integer
month_column = df["month"]
month_column_int = month_column.apply(monthToNum)
df["month"] = month_column_int

#Convert day column to 0 if work day or 1 if week day
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


#SVD
Y = X - np.ones((len(y), 1)) * X.mean(0)
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T

#Projection
Z = Y @ V

j = 0
k = 1
title("PCA1")
plot(Z[:, j], y, "o")
f = figure()
title("PCA2")
plot(Z[:, k], y, "o")
show()
