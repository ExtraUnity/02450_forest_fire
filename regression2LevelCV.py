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


#X = df.drop("area", axis='columns')
#X = df.values
#print(X)
X = df.drop(["area"], axis=1).values

attributeNames = ["X", "Y", "Month",
                  "Day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
### REGRESSION PART B ###

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
max_iter = 10000 


# Define hyperparameters to test
#hidden_units_options = [1, 10, 20, 30, 40, 60, 80]  # Example values
#hidden_units_options = [20, 30, 40, 60, 80, 100, 140, 160, 200] 
hidden_units_options = [1, 10, 15, 20, 25, 30]

MSEs = np.empty(K_outer)
hidden_units = np.empty(K_outer)

for i, (train_index_outer, test_index_outer) in enumerate(outer_cv.split(X, y)):
    # Split data
    X_train_outer, X_test_outer = X[train_index_outer], X[test_index_outer]
    y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]
    
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
    # Retraining and evaluation code here...
    X_train_outer, X_test_outer = torch.Tensor(X_train_outer), torch.Tensor(X_test_outer)
    y_train_outer, y_test_outer = torch.Tensor(y_train_outer), torch.Tensor(y_test_outer)
    
    #Evaluate the model
    y_test_est = best_inner_net(X_test_outer).squeeze()
    
    # Compute the MSE
    #se = (y_test_est.float() - y_test_outer.float())**2
    #mse_outer = (sum(se).type(torch.float)/len(y_test_outer)).data.numpy()
    mse_outer = mean_squared_error(y_test_outer.detach().numpy(), y_test_est.detach().numpy())
    
    # Print the mse_outer with the best hyperparameter
    #print(f"Best hyperparameter: {best_n_units} with MSE: {mse_outer}")
    
    # Store the MSE and the best hyperparameter
    MSEs[i] = mse_outer
    hidden_units[i] = best_n_units
    
    #Print the MSE and the best hyperparameter
    print(f"### FOLD COMPLETED ### MSE: {mse_outer} with {best_n_units} hidden units")
    
    
    
    
# Print the MSEs and the best hyperparameters
print(MSEs)
print(hidden_units)
    
    
    
    

"""
# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize structures to store results
table_results = []
"""
