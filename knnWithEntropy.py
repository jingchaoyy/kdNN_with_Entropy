""" knn ranked with Entropy
Author: Jingchao Yang
Date: Apr.2 2018
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import randomLocations
import entropy
import time

tStart = time.time()

""" Entropy enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
and adjust the the sequence for the last two neighbor based on the entropy value 
The final result is a balance between high diversity and distance
"""


def knn(pS, fTs, pLatLon, k):
    X = np.array(pS)
    pLocation = pS.index(pLatLon) # location for the target point in the list
    neighborsAfter = []  # Storing neighbors after each time getting neighbor check and switched
    # looking for knn one by one
    for i in range(len(pS) + 1):
        if i > 0:  # at least one nearest neighbor (i.e. itself)
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points)
            distances, neighbors = nbrs.kneighbors(X)
            # retrieving neighbors for target point
            targetPNbrs = neighbors[pLocation]

            # adding the latest neighbor to the adjusted neighbor list
            neighborsAfter.append(targetPNbrs[len(targetPNbrs) - 1:len(targetPNbrs)][0])
            print('\nOriginal', assignFT(fTs, neighborsAfter))

            # when more than 2 neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(neighborsAfter) > 2:
                neighborsAfter = checkNeighbor(fTs, neighborsAfter)
                print('Adjusted', assignFT(fTs, neighborsAfter))
            else:  # when less than 2 neighbors found, add to the neighbor list directly
                neighborsAfter = neighborsAfter

    print('\n\n######################## Original Vs. Final Results #################################')
    print('Original Neighbors:', targetPNbrs)
    print('Original Food Type Rank:', assignFT(fTs, targetPNbrs))
    print('Final Neighbors:', neighborsAfter)
    print('Final Food Type Rank:', assignFT(fTs, neighborsAfter))
    return neighborsAfter


""" Function defined to output all neighbors with 
assigned point colors for input point latlon
"""


def assignFT(fTs, nbors):
    neighborTypes = []
    # assign each neighbor with their color type
    for n in nbors:
        neighborTypes.append(fTs[n])
    return neighborTypes


""" Function defined to check if a switch required between tha last two neighbor
based on the entropy calculated
"""


def checkNeighbor(fTs, nbors):
    # assign food type to all the neighbors first
    knnT = assignFT(fTs, nbors)
    # calculating the diversity without the newly added neighbor
    diversity1 = entropy.calcShannonEnt(knnT[:len(knnT) - 1])
    # calculating the diversity if replace the last added neighbor with the newly added neighbor
    diversity2 = entropy.calcShannonEnt(knnT[:len(knnT) - 2] + knnT[len(knnT) - 1:len(knnT)])

    # switching the last two neighbors if adding the later added neighbor first can give a higher entropy
    if diversity2 > diversity1:
        nbors = nbors[:len(nbors) - 2] + nbors[len(nbors) - 1:len(nbors)] + nbors[len(nbors) - 2:len(nbors) - 1]
    else:
        nbors
    return nbors


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

tEnd = time.time()
print("\nTotal time: ", tEnd - tStart, "seconds")
