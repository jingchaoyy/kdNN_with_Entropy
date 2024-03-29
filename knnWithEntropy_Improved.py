""" knn ranked with Entropy (Improved version)
Author: Jingchao Yang
Date: Apr.17 2018
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import randomLocations
import entropy
import time

tStart = time.time()

""" Entropy enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
and adjust the the sequence to compare all combinations in the list to have the max 
entropy sets. In a way, it is always give the most diverse neighbor sets with lowest
total distance
"""


def knn(pS, fTs, pLatLon, k):
    X = np.array(pS)
    pLocation = pS.index(pLatLon)  # location for the target point in the list
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
            # print('\nOriginal', assignFT(fTs, neighborsAfter))

            # when more than 2 neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(neighborsAfter) > k:
                neighborsAfter = checkNeighbor(fTs, neighborsAfter)
                # print('Adjusted', assignFT(fTs, neighborsAfter))
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

    div = []
    for i in range(len(nbors)):
        # computing entropy backwards, and collecting entropy values
        diversity = entropy.calcShannonEnt(knnT[:len(knnT) - i - 1] + knnT[len(knnT) - i:len(knnT)])
        div.append(diversity)
    # print(div)

    # looking for the max entropy
    bestDiv = max(div)
    # getting index of the best neighbor sets, even if there are more then one record matches the max entropy
    # the reason is that the entropy value added to the div list with a sequence that minimize the distance
    # meaning the one in the front has lower total distance
    bestIndex = div.index(bestDiv)
    # print(bestIndex)
    bestNbor = knnT[:len(knnT) - bestIndex - 1] + knnT[len(knnT) - bestIndex:len(knnT)]
    # print('###',bestNbor)

    nbors = nbors[:len(nbors) - bestIndex - 1] + nbors[len(nbors) - bestIndex:len(nbors)]

    return nbors


if __name__ == "__main__":
    # generating 100 points randomly
    pSets = randomLocations.genPoints(100)
    # print(pSets)
    fTypes = randomLocations.fType(pSets)
    # print(fTypes)

    # calculating knn for each point
    # e.g. find all neighbors with types for the first point
    p0 = 0
    k = 6  # Num of neighbors
    neighbors = knn(pSets, fTypes, pSets[p0], k)
    # print(neighbors)

tEnd = time.time()
print("\nTotal time: ", tEnd - tStart, "seconds")
