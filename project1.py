# exercise 2.1.1
import math
import importlib_resources
import numpy as np
import pandas as pd
from utils import monthToNum, dayToNum
from scipy.linalg import svd
from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel
import pylab as py 
import matplotlib.pyplot as plt

# Load xls sheet with data
df = pd.read_csv("forestfires.csv")

#Log transform the area
y_t = df["area"]

y = (y_t+1).apply(math.log)
#df["area"] = y
py.show()

plt.hist(y_t)
show()
plt.hist(y)
show()
#Convert month column to integer
month_column = df["month"]
month_column_int = month_column.apply(monthToNum)
df["month"] = month_column_int

#Convert day column to 0 if work day or 1 if weekend
day_column = df["day"]
day_column_int = day_column.apply(dayToNum)
df["day"] = day_column_int


X = df.loc[:, df.columns != "month"].values
y_m = df["month"]



#One-out-of-K encoding for month
month = np.array(df["month"].values, dtype=int).T
K = month.max()
month_encoding = np.zeros((month.size, K))
month_encoding[np.arange(month.size), month-1] = 1
X = np.concatenate((X[:,:-1], month_encoding), axis=1)

#SVD
#Standardize the data
Y = (X - np.ones((len(y), 1)) * X.mean(0)) / X.std(0)
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T

#Projection
Z = Y @ V

j = 0
k = 1

#Plot pc1 and pc2 for log_area
for c in range(5):
    # select indices belonging to class c:
    class_mask1 = y > (c-1)/2
    class_mask2 = y <= c/2
    class_mask = class_mask1.values & class_mask2.values
    plot(Z[class_mask, j], Z[class_mask, k], "o")
legend(["0", "]0;0.5]", "]0.5;1]", "]1;1.5]", "1.5<"])
title("Log(area+1)")
xlabel("PC1")
ylabel("PC2")
show()

#Plot pc1 and pc2 for months
for c in range(12):
    # select indices belonging to class c:
    class_mask = df["month"] == c
    plot(Z[class_mask, j], Z[class_mask, k], "o", label="{0}".format(c+1))
legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
title("Month")
xlabel("PC1")
ylabel("PC2")
show()

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()

