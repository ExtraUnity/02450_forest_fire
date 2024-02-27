import math
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read CSV file.
df = pd.read_csv(r"C:\Users\rosam\Documents\DTU\02450 Machine Learning and Data Mining\Project 1 Forest Fires\forestfires.csv")
#print(df)

### SUMMARY STATISTICS
summary_df = df.describe()
print(summary_df)
#skewness = df.skew()
#additional_stats = pd.DataFrame([skewness], index = ['skewness'])

#summary_stats = pd.concat([summary_df, additional_stats])
#print(summary_stats)

### CORRELATION ANALYSIS AND PLOT
# log-transform area
df['area']=(df['area']+2).apply(math.log)
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
plt.hist2d(df['X'], df['Y'], bins=[np.arange(min(df['X'])-0.5, max(df['X'])+1.5, 1), np.arange(min(df['Y'])-0.5, max(df['Y'])+1.5, 1)], cmap='GnYlRd')
plt.colorbar(label='Number of Fires')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Fire Occurrences')
plt.grid(True)
plt.show()

"""
# Generate a histogram for each attribute in the dataset
for column in df.columns:
    plt.figure(figsize=(10, 4))
    df[column].hist(bins=10)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()


plt.hist(df['DMC'],bins=10,alpha=0.5,histtype="bar",density=True,ec="black")
plt.title('DMC')
plt.show()

X = np.array(df.iloc[:,0])
Y = np.array(df.iloc[:,1])
month = np.array(df.iloc[:,2])
day = np.array(df.iloc[:,3])
FFMC = np.array(df.iloc[:,4])
DMC = np.array(df.iloc[:,5])
DC = np.array(df.iloc[:,6])
ISI = np.array(df.iloc[:,7])
temp = np.array(df.iloc[:,8])
RH = np.array(df.iloc[:,9])
wind = np.array(df.iloc[:,10])
rain = np.array(df.iloc[:,11])
area = np.array(df.iloc[:,12])


#plt.hist(DMC,bins=10,density=True)
#plt.show()
"""
