# Self-Organizing Map using Air Cargo Routes
## Introduction
This repository demonstrates the application of a Self-Organizing Map (SOM) using the air cargo routes dataset. SOM is an unsupervised neural network technique that clusters data and visualizes high-dimensional datasets into lower dimensions.
## Features
- Load and preprocess air cargo route data
- Apply a Self-Organizing Map to find clusters of similar routes
- Visualize the resulting map using a U-Matrix (Unified Distance Matrix)
- Explore route similarities and clusters using SOM
## Dataset
The dataset contains air cargo routes with details such as:
- `route_id`: Unique identifier for each route
- `flight_num`: Flight number for the cargo route
- `origin_airport`: Airport code of origin
- `destination_airport`: Airport code of destination
- `aircraft_id`: Type of aircraft used
- `distance_miles`: Distance between the origin and destination
## Getting Started
Prerequisites      
Python 3.7 or later
Libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `MiniSom`
Install the required dependencies using:
```bash
pip install -r requirements.txt
```
## Dataset
Make sure the dataset (`air_cargo_routes.csv`) is in the root directory of the project. The dataset contains columns such as route ID, flight number, origin and destination airport codes, aircraft type, and route distance in miles.
## Installation
1. Clone the repository:
```bash
git clone https://github.com/Shy810/Self-Organizing-map-using-air-cargo-routes-.git
cd Self-Organizing-map-using-air-cargo-routes-
```
2. Install the dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Python script:
```bash
python som_air_cargo.py
```
This will load the dataset, apply the SOM, and display a visualization of the clusters.
## Code Explanation
1. **Data Loading and Preprocessing**: - Load the CSV file containing air cargo route data.
- Normalize the selected features (`flight_num`, `aircraft_id`, and `distance_miles`) using `MinMaxScaler`.
2. **SOM Model Creation**: - A Self-Organizing Map is created using the `MiniSom` library with a 10x10 grid of neurons. The SOM algorithm is trained on the normalized data.
3. **Visualization**: - The U-Matrix (Unified Distance Matrix) is visualized to show clusters of similar cargo routes. Each route is labeled on the SOM based on the `flight_num`.
## Output
The output will be a visualization of the SOM where each node represents a cluster of similar air cargo routes. The distance map (U-Matrix) highlights the relationships between routes based on the data.
