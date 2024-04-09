import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np

# Load the map image
map_img = mpimg.imread('Figures/mapOfForest.png')
map_img = np.fliplr(map_img)

# Load the forest fire data
df = pd.read_csv('forestfires.csv')

# Extract X and Y coordinates
x = df['X'].values
y = df['Y'].values

# Define the bins for the histogram
x_bins = np.arange(min(x)-0.5, max(x)+1.5, 1)
y_bins = np.arange(min(y)-0.5, max(y)+1.5, 1)

# Create a 2D histogram / heatmap data
heatmap_data, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

# Plot the map
plt.figure(figsize=(10, 8))
plt.imshow(map_img, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='upper')

# Overlay the heatmap
# The alpha argument controls the transparency of the heatmap
plt.imshow(heatmap_data.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot', alpha=0.5, interpolation='nearest')

# Add color bar and other plot elements
plt.colorbar(label='Number of Fires')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Fire Occurrences on Forest Map')
plt.grid(True)

# Show the plot
plt.show()
