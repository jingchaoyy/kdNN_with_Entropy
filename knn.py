""" collecting k nearest neighbors
Author: Jingchao Yang
Date: Apr.2 2018
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import randomLocations

# generating 100 points randomly
points = randomLocations.genPoints(1000)
X = np.array(points)

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
# return distances and ranked neighbors
distances, indices = nbrs.kneighbors(X)
print(indices)