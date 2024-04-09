import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Improve the aesthetics with seaborn
sns.set(style="whitegrid")

# Load the dataset
forest_fires_data = pd.read_csv('forestfires.csv')  # Update this path

# Selecting only numerical columns for histograms
numerical_columns = forest_fires_data.select_dtypes(include=['float64', 'int64']).columns

# Setting up the figure size and layout
plt.figure(figsize=(20, 15))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)  # Adjust grid size accordingly
    sns.histplot(forest_fires_data[column], kde=False, color='skyblue', bins='auto', alpha=0.7)
    plt.title(column, fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=3.0)

# Overall plot adjustments
#plt.suptitle('Histograms of Forest Fire Data Attributes', fontsize=16, y=1.02)
plt.show()
