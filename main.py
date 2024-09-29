import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the CSV Data
data = pd.read_csv('air_cargo_routes.csv')  # Replace with your file name

# Step 2: Data Preprocessing
# Select relevant features for SOM training (e.g., 'distance_miles', etc.)
features = data[['distance_miles']].values  # You can add more features if needed

# Normalize the data (scale to [0, 1])
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Create and Train the SOM
# Define the size of the SOM grid (e.g., 10x10)
som_size = (10, 10)
som = MiniSom(som_size[0], som_size[1], features_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Initialize the weights and train the SOM
som.random_weights_init(features_scaled)
som.train_random(features_scaled, num_iteration=1000)

# Step 4: Visualize the Results
# Plot the SOM grid with the distance of the nodes
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Plot the distance map
plt.colorbar()

# Optionally, add markers for the input data points
markers = ['o', 's', 'D', '^']
colors = ['r', 'g', 'b', 'y']
for i, x in enumerate(features_scaled):
    w = som.winner(x)
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[data['route_id'][i] % len(markers)], 
             markerfacecolor='None', markeredgecolor=colors[data['route_id'][i] % len(colors)], markersize=10, markeredgewidth=2)

plt.title('SOM - Air Cargo Routes Clustering')
plt.show()
