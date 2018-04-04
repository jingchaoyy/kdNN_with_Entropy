""" knn ranked with Entropy
Author: Jingchao Yang
Date: Apr.2 2018
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import randomLocations
import entropy

# simple knn

def knn(pS, fT, pLatLon, k):
    X = np.array(pS)
    for i in range(k+1):
        if i > 0:
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points
            distances, neighbors = nbrs.kneighbors(X)
            nTypes = assignColor(pLatLon, pS, fT, neighbors)






    return neighbors,nTypes


""" Function defined to output all neighbors with 
assigned point colors for input point latlon
"""
def assignColor(pLatLon, pS, fT, nbors):
    # check if point exits
    if pLatLon in pS:
        # find the point location
        pLocation = pS.index(pLatLon)
        # getting all the neighbors
        pNeighbors = nbors[pLocation]
        neighborTypes = []
        # assign each neighbor with their color type
        for n in pNeighbors:
            neighborTypes.append(fT[n])
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
    # e.g. find all neighbors with types for the first point
    p0 = 0
    k = 5
    neighborTypes = knn(pSets,fTypes,pSets[p0],k)
    print(neighborTypes)

