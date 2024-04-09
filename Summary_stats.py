import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read CSV file.
df = pd.read_csv("forestfires.csv")
print(df)

### SUMMARY STATISTICS
summary_df = df.describe()
print(summary_df)

### CORRELATION ANALYSIS AND PLOT
# log-transform area
df['area']=(df['area']+1).apply(math.log)
# Select only the numeric attributes relevant for correlation analysis
numeric_df = df[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]


# Compute the pairwise Pearson correlation matrix
correlation_matrix = numeric_df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Matrix of Numeric Attributes")
plt.show()


### Distribution of fires, plot
# Create a 2D histogram / heatmap of fire occurrences based on X and Y coordinates
plt.figure(figsize=(10, 8))
plt.hist2d(df['X'], df['Y'], bins=[np.arange(min(df['X'])-0.5, max(df['X'])+1.5, 1), np.arange(min(df['Y'])-0.5, max(df['Y'])+1.5, 1)], cmap='RdYlGn_r')
plt.colorbar(label='Number of Fires')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Fire Occurrences')
plt.grid(True)
plt.show()
