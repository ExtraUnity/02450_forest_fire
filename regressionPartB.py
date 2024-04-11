import math
import importlib_resources
import numpy as np
import pandas as pd
from utils import monthToNum, dayToNum
from scipy.linalg import svd
from matplotlib.pyplot import figure, legend, plot, show, title, xlabel, ylabel
import matplotlib.pyplot as plt

    
import torch
from torch import nn, optim
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_iris
from dtuimldmtools import draw_neural_net, train_neural_net, rlr_validate

# Load xls sheet with data
df = pd.read_csv("forestfires.csv")

#Log transform the area
y_t = df["area"]

y = (y_t+1).apply(math.log)
df["area"] = y
y = y.to_numpy()
y = y.reshape(517, 1)


# Before transforming and one-hot encoding, extract attribute names
attributeNames = list(df.columns)
attributeNames.remove('month')  # 'month' will be one-hot encoded, so remove it
attributeNames.remove('area')  # Assuming 'area' is the target variable and not included in X

#Convert month column to integer
month_column = df["month"]
month_column_int = month_column.apply(monthToNum)
df["month"] = month_column_int

#Convert day column to 0 if work day or 1 if weekend
day_column = df["day"]
day_column_int = day_column.apply(dayToNum)
df["day"] = day_column_int

#print(df.to_string())

### Reg par A ###
X_rlr = df.loc[:, df.columns != "month"].values
#One-out-of-K encoding for month
month = np.array(df["month"].values, dtype=int).T
K = month.max()
month_encoding = np.zeros((month.size, K))
month_encoding[np.arange(month.size), month-1] = 1
X_rlr = np.concatenate((X_rlr[:,:-1], month_encoding), axis=1)

N, M = X_rlr.shape

#Standardize the data
X_rlr = (X_rlr - np.ones((len(y), 1)) * X_rlr.mean(0)) / X_rlr.std(0)

# Add offset attribute
X_rlr = np.concatenate((np.ones((X_rlr.shape[0], 1)), X_rlr), 1)
#attributeNames = ["Offset"] + attributeNames
M = M + 1

#lambdas = np.power(10.0, [2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5])
lambdas = np.power(10.0, [1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5])
####################################################



### REGRESSION PART B ###
X = df.drop(["area"], axis=1).values
# method to create multiple ANN's
def create_ann_model(input_dim, n_hidden_units):
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(input_dim, n_hidden_units),  # Input layer to hidden layer
        torch.nn.Tanh(),                             # Activation function
        torch.nn.Linear(n_hidden_units, 1)           # Hidden layer to output layer
    )
    return model


seed = 2

K_outer = 10
K_inner = 10

## ANN ##
# Initialize KFold cross-validation objects
outer_cv = model_selection.KFold(n_splits=K_outer, shuffle=True, random_state=seed)
inner_cv = model_selection.KFold(n_splits=K_inner, shuffle=True, random_state=seed)

# Parameters for neural network classifier
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 1000


# Define hyperparameters to test
#hidden_units_options = [1, 10, 20, 30, 40, 60, 80]  # Example values
#hidden_units_options = [20, 30, 40, 60, 80, 100, 140, 160, 200] 
hidden_units_options = [1, 10, 15, 20, 25, 30]
#hidden_units_options = [1]

MSEs = np.empty(K_outer)
hidden_units = np.empty(K_outer)
found_lambdas = np.empty(K_outer)
found_rlr_errors = np.empty(K_outer)
baseline_errors = np.empty(K_outer)

for i, (train_index_outer, test_index_outer) in enumerate(outer_cv.split(X, y)):
    # Split data
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]
    
    # Sptlits for rlr
    X_train_outer_rlr, X_test_outer_rlr = X_rlr[train_index_outer], X_rlr[test_index_outer]
    y_train_outer_rlr, y_test_outer_rlr = y[train_index_outer], y[test_index_outer]
    
    # Inner CV for hyperparameter tuning
    
    best_inner_error = np.inf
    best_inner_net = None
    best_n_units = None
        
    for j, (train_index_inner, test_index_inner) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
        # Prepare data
        X_train_inner, X_test_inner = torch.Tensor(X_train_outer[train_index_inner]), torch.Tensor(X_train_outer[test_index_inner])
        y_train_inner, y_test_inner = torch.Tensor(y_train_outer[train_index_inner]), torch.Tensor(y_train_outer[test_index_inner])

        for n_hidden_units in hidden_units_options:
            print(f"Training with {n_hidden_units} hidden units")
            
            
            # Create and train model
            model = create_ann_model((X.shape[1]), n_hidden_units)
            loss_fn = nn.MSELoss()
            #optimizer = optim.Adam(model.parameters())
            
            # Training code here...
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X = X_train_inner,
                y = y_train_inner,
                n_replicates = n_replicates,
                max_iter = max_iter
            )

            print(f"Final loss: {final_loss}")
            
            # Store final_loss for this hyperparameter if smallest
            if final_loss < best_inner_error:
                best_inner_net = net
                best_n_units = n_hidden_units
                best_inner_error = final_loss
                print(f"New best hyperparameter: {best_n_units} with MSE: {best_inner_error}")
                
                   
    
    # Retrain using the best hyperparameter on the entire outer training set
    # Retraining and evaluation code
    X_train_outer, X_test_outer = torch.Tensor(X_train_outer), torch.Tensor(X_test_outer)
    y_train_outer, y_test_outer = torch.Tensor(y_train_outer), torch.Tensor(y_test_outer)
    
    #Evaluate the model
    y_test_est = best_inner_net(X_test_outer).squeeze()
    
    # Compute the MSE
    mse_outer = mean_squared_error(y_test_outer.detach().numpy(), y_test_est.detach().numpy())

    # Store the MSE and the best hyperparameter
    MSEs[i] = mse_outer
    hidden_units[i] = best_n_units
    
    #Print the MSE and the best hyperparameter
    print(f"### FOLD COMPLETED ### MSE: {mse_outer} with {best_n_units} hidden units")
    
    ### RLR ###
    (
        opt_val_err,
        opt_lambda,
        mean_w_vs_lambda,
        train_err_vs_lambda,
        test_err_vs_lambda,
    ) = rlr_validate(X_train_outer_rlr, y_train_outer_rlr, lambdas, K_outer)
    
    found_lambdas[i] = opt_lambda
    found_rlr_errors[i] = opt_val_err
    
    ### Baseline ###
    # Baseline Model: Linear regression with no features (mean prediction)
    y_mean = torch.mean(y_train_outer).item()
    y_pred_baseline = np.full(y_test_outer.shape, y_mean)
    mse_baseline = mean_squared_error(y_test_outer, y_pred_baseline)
    baseline_errors[i] = mse_baseline
    
    
    
# Print the MSEs and the best hyperparameters
"""
print(MSEs)
print(hidden_units)
print(found_lambdas)
print(found_rlr_errors)
print(baseline_errors)
"""   

print(f"{'#'*40}")
print(f"{' Model Performance Summary ':^40}")
print(f"{'#'*40}")

# Formatting arrays for a cleaner display
mse_strings = ', '.join([f"{mse:.4f}" for mse in MSEs])
hidden_unit_strings = ', '.join([str(int(unit)) for unit in hidden_units])
found_lambdas_strings = ', '.join([f"{lmbda:.2e}" for lmbda in found_lambdas])
found_rlr_errors_strings = ', '.join([f"{error:.4f}" for error in found_rlr_errors])
baseline_error_strings = ', '.join([f"{error:.4f}" for error in baseline_errors])

# Printing the formatted strings
print(f"MSEs across folds:\n[{mse_strings}]")
print(f"Optimal hidden units per fold:\n[{hidden_unit_strings}]")
print(f"Optimal lambdas per fold:\n[{found_lambdas_strings}]")
print(f"RLR Validation Errors per fold:\n[{found_rlr_errors_strings}]")
print(f"Baseline Model Errors per fold:\n[{baseline_error_strings}]")
print(f"{'-'*40}")

    
from scipy.stats import ttest_rel, t, sem
# Paired t-test between ANN and linear regression
t_stat_ann_lr, p_val_ann_lr = ttest_rel(MSEs, found_rlr_errors)
print(f"ANN vs. Linear Regression: t-statistic = {t_stat_ann_lr}, p-value = {p_val_ann_lr}")

# Paired t-test between ANN and baseline
t_stat_ann_baseline, p_val_ann_baseline = ttest_rel(MSEs, baseline_errors)
print(f"ANN vs. Baseline: t-statistic = {t_stat_ann_baseline}, p-value = {p_val_ann_baseline}")

# Paired t-test between linear regression and baseline
t_stat_lr_baseline, p_val_lr_baseline = ttest_rel(found_rlr_errors, baseline_errors)
print(f"Linear Regression vs. Baseline: t-statistic = {t_stat_lr_baseline}, p-value = {p_val_lr_baseline}")

def calculate_confidence_interval(data1, data2, confidence=0.95):
    # Calculate the differences
    differences = np.array(data1) - np.array(data2)
    
    # Mean of differences
    diff_mean = np.mean(differences)
    
    # Standard error of the mean of differences
    std_err = sem(differences)
    
    # Degrees of freedom
    dof = len(differences) - 1
    
    # Critical value from the t-distribution
    t_crit = t.ppf((1 + confidence) / 2, dof)
    
    # Margin of error
    margin_error = t_crit * std_err
    
    # Confidence interval
    conf_interval = (diff_mean - margin_error, diff_mean + margin_error)
    
    return conf_interval

ci_ann_vs_lr = calculate_confidence_interval(MSEs, found_rlr_errors)
ci_ann_vs_baseline = calculate_confidence_interval(MSEs, baseline_errors)
ci_lr_vs_baseline = calculate_confidence_interval(found_rlr_errors, baseline_errors)

print(f"95% CI for the difference in means (ANN vs. LR): {ci_ann_vs_lr}")
print(f"95% CI for the difference in means (ANN vs. Baseline): {ci_ann_vs_baseline}")
print(f"95% CI for the difference in means (LR vs. Baseline): {ci_lr_vs_baseline}")



