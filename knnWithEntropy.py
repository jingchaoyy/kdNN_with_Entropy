""" knn ranked with Entropy
Author: Jingchao Yang
Date: Apr.2 2018
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
import randomLocations
import entropy


# simple knn

def knn(pS, fTs, pLatLon, k):
    X = np.array(pS)
    diversities = []  # storing diversity each time adding a new neighbor
    kNNs = []
    neighborsAfter = []
    # looking for knn one by one
    for i in range(len(pS) + 1):
        if i > 0:  # at least one nearest neighbor (i.e. itself)
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points)
            distances, neighbors = nbrs.kneighbors(X)
            nbrTys = assignColor(pLatLon, pS, fTs, neighbors)
            print('\nOriginal',nbrTys)
            diversity = entropy.calcShannonEnt(nbrTys)
            # print(diversity)
            diversities.append(diversity)
            kNNs.append(nbrTys)
            if len(kNNs) > 2:
                neighborsAfter = checkNeighbor(nbrTys)

    return neighbors, nbrTys


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


def checkNeighbor(knns):
    # calculating the diversity without the newly added neighbor
    diversity1 = entropy.calcShannonEnt(knns[:len(knns) - 1])
    # calculating the diversity if replace the last added neighbor with the newly added neighbor
    diversity2 = entropy.calcShannonEnt(knns[:len(knns) - 2] + knns[len(knns) - 1:len(knns)])
    # switching the last two neighbors since this can give a higher entropy value for k-1
    if diversity2 > diversity1:
        knns = knns[:len(knns) - 2] + knns[len(knns) - 1:len(knns)] + knns[len(knns) - 2:len(knns) - 1]
    else:
        knns
    print(knns)
    return knns


if __name__ == "__main__":
    # generating 100 points randomly
    pSets = randomLocations.genPoints(10)
    # print(pSets)
    fTypes = randomLocations.fType(pSets)
    # print(fTypes)

    # calculating knn for each point
    # e.g. find all neighbors with types for the first point
    p0 = 0
    k = 5
    neighborTypes = knn(pSets, fTypes, pSets[p0], k)
    # print(neighborTypes)