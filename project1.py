# exercise 2.1.1
import importlib_resources
import numpy as np
import pandas as pd

def monthToNum(shortMonth):
    return {
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9, 
            'oct': 10,
            'nov': 11,
            'dec': 12
    }[shortMonth]

def dayToNum(day):
        return {
            'mon': 0,
            'tue': 0,
            'wed': 0,
            'thu': 0,
            'fri': 1,
            'sat': 1,
            'sun': 1,
    }[day]

# Load xls sheet with data
df = pd.read_csv("forestfires.csv")
y = df["area"]

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
X = np.concatenate((X[:, :-1], month_encoding), axis=1)
print(X[0])