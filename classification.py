import pandas as pd
import numpy as np
import math as math
from sklearn import model_selection, tree
import sklearn.linear_model as lm
from utils import monthToNum, dayToNum
from dtuimldmtools import mcnemar

filename = 'forestfires.csv'
df = pd.read_csv(filename)
# Log transform the area
y_t = df["area"]
y = (y_t+1).apply(math.log)
y_median = y.median()
y = np.asarray(y > 0)
# Convert month column to integer
month_column = df["month"]
month_column_int = month_column.apply(monthToNum)
df["month"] = month_column_int

# Convert day column to 0 if work day or 1 if weekend
day_column = df["day"]
day_column_int = day_column.apply(dayToNum)
df["day"] = day_column_int


X = df.loc[:, df.columns != "area"].values


attributeNames = ["X", "Y", "Month",
                  "Day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]

classLabels = np.array([1 if b else 0 for b in y])
y = classLabels
K = len(classLabels)
classNames = (set(classLabels))
classDict = dict(zip(classNames, range(K)))

yr = X[:,-1].astype(float).squeeze()
yc = np.asarray([classDict[value] for value in classLabels],dtype=float)
N = len(yr)
M = len(attributeNames)
C = len(classNames) 
#  Add offset attribute
X = (X-X.mean(0))/X.std(0)
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M = M + 1
seed = 3

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.0, range(-5, 9))

# Depths for decision tree
tc = np.arange(1, 11, 1, dtype=int)

# Initialize variables
Error_train_log = np.empty((K, 1))
Error_test_log = np.empty((K, 1))
Error_train_tree = np.empty((K, 1))
Error_test_tree = np.empty((K, 1))
Error_test_baseline = np.empty((K, 1))
opt_lambda = np.zeros(K)
opt_depth = np.zeros(K)
y_true_values = []
log_y_est = []
tree_y_est = []
baseline_y_est = []

print("Outer fold   Logistic Regression     Classification tree     Baseline")
print("i            opt_lambda  Error       max_depth   Error       Error")
for j, (train_index, test_index) in enumerate(CV.split(X, y)):
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10
    CV2 = model_selection.KFold(internal_cross_validation, shuffle=True)
    y_true_values.append(y_test)

    train_error_inner_log = np.empty((internal_cross_validation, len(lambdas)))
    test_error_inner_log = np.empty((internal_cross_validation, len(lambdas)))

    train_error_inner_tree = np.empty((internal_cross_validation, len(tc)))
    test_error_inner_tree = np.empty((internal_cross_validation, len(tc)))

    test_error_inner_baseline = np.empty((internal_cross_validation,1))

    # INNER FOLD
    for i, (train_index_inner, test_index_inner) in enumerate(CV.split(X_train, y_train)):
        X_train_inner = X_train[train_index_inner,:]
        y_train_inner = y_train[train_index_inner]
        X_test_inner = X_train[test_index_inner]
        y_test_inner = y_train[test_index_inner]

        # TRAIN LOGISTIC REGRESSION FOR ALL LAMBDA VALUES
        for k in range(len(lambdas)):
            mdl = lm.LogisticRegression(penalty='l2', C=1/lambdas[k] )
    
            mdl.fit(X_train_inner, y_train_inner)

            y_train_est = mdl.predict(X_train_inner).T
            y_test_est = mdl.predict(X_test_inner).T

            train_error_inner_log[i,k] = np.sum(y_train_est != y_train_inner) / len(y_train_inner)
            test_error_inner_log[i,k] = np.sum(y_test_est != y_test_inner) / len(y_test_inner)


        # TRAIN CT FOR ALL DEPTHS
        for k, t in enumerate(tc):
            mdl = tree.DecisionTreeClassifier(criterion="gini", max_depth=int(t))

            mdl.fit(X_train_inner, y_train_inner)

            y_train_est = mdl.predict(X_train_inner).T
            y_test_est = mdl.predict(X_test_inner).T

            train_error_inner_tree[i,k] = np.sum(y_train_est != y_train_inner) / len(y_train_inner)
            test_error_inner_tree[i,k] = np.sum(y_test_est != y_test_inner) / len(y_test_inner)


        # BASELINE MODEL
        y_test_est = np.ones(len(y_test_inner)) if np.sum(X_train_inner) > len(X_train_inner) else np.zeros(len(y_test_inner))
        test_error_inner_baseline[i,0] = np.sum(y_test_est != y_test_inner) / len(y_test_inner)

    ## LOGISTIC REGRESSION
    # CHOOSE OPTIMAL LAMBDA VALUE AND FIT MODEL
    opt_lambda[j] = lambdas[np.argmin(test_error_inner_log[i])]
    mdl_log = lm.LogisticRegression(penalty='l2', C=1/opt_lambda[j])
    mdl_log.fit(X_train, y_train)

    # ESTIMATE y USING BEST MODEL
    y_train_est_log = mdl_log.predict(X_train).T
    y_test_est_log = mdl_log.predict(X_test).T

    # COMPUTE ERROR FOR LOGISTIC REGRESSION
    Error_train_log[j] = np.sum(y_train_est_log != y_train) / len(y_train)
    Error_test_log[j] = np.sum(y_test_est_log != y_test) / len(y_test)

    log_y_est.append(y_test_est_log)

    ## DECISION TREE
    # CHOOSE OPTIMAL DEPTH AND FIT MODEL
    opt_depth[j] = tc[np.argmin(test_error_inner_tree[i])]
    mdl_tree = tree.DecisionTreeClassifier(criterion="gini", max_depth=int(opt_depth[j]))
    mdl_tree.fit(X_train, y_train)

    # ESTIMATE y USING BEST MODEL
    y_train_est_tree = mdl_tree.predict(X_train).T
    y_test_est_tree = mdl_tree.predict(X_test).T

    # COMPUTE ERROR FOR DECISION TREE
    Error_train_tree[j] = np.sum(y_train_est_tree != y_train) / len(y_train)
    Error_test_tree[j] = np.sum(y_test_est_tree != y_test) / len(y_test)

    tree_y_est.append(y_test_est_tree)

    ## BASELINE MODEL
    y_test_est_baseline = np.ones(len(y_test)) if np.sum(classLabels) > len(classLabels) else np.zeros(len(y_test))
    Error_test_baseline[j] = np.sum(y_test_est_baseline != y_test) / len(y_test)

    baseline_y_est.append(y_test_est_baseline)

    print(str(j) + "            " + str(opt_lambda[j]) + "   "+str(Error_test_log[j][0]) + "     "+ str(opt_depth[j]) + "     " + str(Error_test_tree[j][0]) + "    " + str(Error_test_baseline[j][0]))

y_true_values = np.concatenate(y_true_values)
log_y_est = np.concatenate(log_y_est)
tree_y_est = np.concatenate(tree_y_est)
baseline_y_est = np.concatenate(baseline_y_est)


mdl_final = lm.LogisticRegression(penalty='l2', C=1/100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.8
)
mdl_final.fit(X_train, y_train)
y_est = mdl_final.predict(X_test).T
error = np.sum(y_est != y_test) / len(y_test)

## Results

print("Final logistic regression model error: " + str(error))
print("Weights:")
print(mdl_final.coef_)
print("Logistic regression generalization error: " + str(Error_test_log.mean()))
print("Classification tree generalization error: " + str(Error_test_tree.mean()))
print("Baseline generalization error: " + str(Error_test_baseline.mean()))


print("-----McNemar's test-----")
print("Logistic vs Tree")
print(mcnemar(y_true=y_true_values, yhatA=log_y_est, yhatB=tree_y_est))
print("Logistic vs Baseline")
print(mcnemar(y_true=y_true_values, yhatA=log_y_est, yhatB=baseline_y_est))
print("Tree vs Baseline")
print(mcnemar(y_true=y_true_values, yhatA=tree_y_est, yhatB=baseline_y_est))
