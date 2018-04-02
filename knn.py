""" collecting k nearest neighbors
Author: Jingchao Yang
Date: Apr.2 2018
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import randomLocations

# simple knn

def knn(p):
    X = np.array(p)
    nbrs = NearestNeighbors(n_neighbors=len(X), algorithm='ball_tree').fit(X)
    # return distances and ranked neighbors (presented as point location in array points
    distances, neighbors = nbrs.kneighbors(X)
    return distances, neighbors

""" Function defined to output all neighbors with 
assigned point colors for input point latlon
"""
def typeRanks(pLatLon, neighbors):
    if pLatLon in points:
        pLocation = points.index(pLatLon)
        pNeighbors = neighbors[pLocation]
        neighborTypes = []
        for n in pNeighbors:
            neighborTypes.append(pTypes[n])
        return neighborTypes
    else:
        print("Point not found")

if __name__ == "__main__":
    # generating 100 points randomly
    points = randomLocations.genPoints(5)
    print(points)
    pTypes = randomLocations.pColor(points)
    print(pTypes)

    # calculating knn for each point
    neighbors = knn(points)[1]
    print(neighbors)
    # e.g. find all neighbors with types for the first point
    print(typeRanks(points[0],neighbors))
