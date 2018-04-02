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
def assignFT(pLatLon, pS, fTs, nbors):
    # check if point exits
    if pLatLon in pS:
        # find the point location
        pLocation = pS.index(pLatLon)
        # getting all the point neighbors
        pNeighbors = nbors[pLocation]
        neighborTypes = []
        # assign each neighbor with their color type
        for n in pNeighbors:
            neighborTypes.append(fTs[n])
        return neighborTypes
    else:
        print("Point not found")

if __name__ == "__main__":
    # generating 100 points randomly
    pSets = randomLocations.genPoints(10)
    print(pSets)
    fTypes = randomLocations.fType(pSets)
    print(fTypes)

    # calculating knn for each point
    neighbors = knn(pSets)[1]
    print(neighbors)
    # e.g. find all neighbors with types for the first point
    p0 = 0
    print(assignFT(pSets[p0],pSets,fTypes,neighbors))
