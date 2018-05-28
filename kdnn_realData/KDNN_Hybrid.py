"""
Created on 5/27/18

@author: Jingchao Yang
"""

from sklearn.neighbors import NearestNeighbors
import numpy as np
from kdnn_realData import yelpDataCollector
from kdnn_realData import entropyGetWeight
import time

tStart = time.time()

""" Entropy enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
Using all categories found within neighbors to calculate entropy weight for each category, 
and use the weighted entropy to calculated the final total entropy for each neighbor,
choose k that ranked as the highest
"""


def knn(pS, fTs, pLatLon, kk):
    X = np.array(pS)
    pLocation = pS.index(pLatLon)  # location for the target point in the list
    nonDominated = []
    targetPNbrs = ''
    for i in range(len(pS) + 1):
        if i > 0:  # at least one nearest neighbor (i.e. itself)
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points)
            distances, neighbors = nbrs.kneighbors(X)
            # retrieving neighbors for target point
            targetPNbrs = neighbors[pLocation]
            print("\nOriginal NID", targetPNbrs)

            # when more than k neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(targetPNbrs) > kk:
                bestDiv = checkNeighbor(fTs, targetPNbrs, kk)
                print('Adjusted NID', bestDiv)
                # print('nondominated neighbor set:', bestDiv)

                if nonDominated[-1] != bestDiv:  # see if the last added nonDominated sets is tha same
                    #  as the latest one, if the same, ignore the latest one
                    nonDominated.append(bestDiv)

            else:  # when less than minimum required neighbors found, add to the neighbor list directly
                # print('nondominated neighbor set:', targetPNbrs)
                if len(targetPNbrs) == kk:  # add the first find knn set to the nonDominated list
                    nonDominated.append(list(targetPNbrs))

    return nonDominated


""" Function defined to output all neighbors with 
assigned food types for input point latlon
"""


def assignFT(fTs, nbors):
    neighborTypes = []
    # assign each neighbor with their color type
    for n in nbors:
        neighborTypes.append(fTs[n])
    return neighborTypes


""" Function defined to gather weight info for each category during 
entropy calculation, rank each restaurant based on the total weighted
entropy, and output k restaurants with highest entropy value
"""


def checkNeighbor(fTs, nbors, kk):
    # assign food type to all the neighbors first
    knnT = assignFT(fTs, nbors)

    ftList = []
    for x in knnT:
        for y in x:
            ftList.append(y)

    weights = entropyGetWeight.calcShannonEnt(ftList)[1]

    div = []  # list for recording total entropy of each neighbor, and related neighbor
    for i in knnT:
        iEntropy = 0  # for recording the entropy of neighbor i
        for j in i:
            for k in weights:
                if j in k:  # assign weight of categories to each category in that neighbor
                    iEntropy += k[1]
        div.append((iEntropy, nbors[knnT.index(i)]))
    print("divOriginal", div)

    div.sort(reverse=True)  # reverse sort list, from largest weighted div to smallest
    print("divAdjusted", div)

    knbors = []
    for z in div:
        knbors.append(z[1])  # collect neighbors
    knbors = knbors[:kk]  # get the first 6, as kk == 6

    neighborList = []  # put the selected neighbors in its original order (distance based)
    for a in nbors:
        for b in knbors:
            if a == b:
                neighborList.append(a)

    return neighborList


"""Main"""

if __name__ == "__main__":
    # generating 100 points randomly
    pSets = yelpDataCollector.allPoints
    # print(pSets[0])
    fTypes = yelpDataCollector.allCategories
    # print(fTypes)

    userAddr = [(35.04728681, -80.99055881)]  # input user location
    addUser = userAddr + pSets[:50]  # add user location to the restaurant list
    fullSet = np.array(addUser)  # convert to numpy array for knn
    nearestRest = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(fullSet)
    dist, restIndex = nearestRest.kneighbors(fullSet)  # knn for k=2
    # retrieving neighbors for target point
    p0 = restIndex[0][1] - 1  # get the nearest restaurant of user,
    # and assign p0 with the restaurant index of original restaurant list

    kk = 6  # Num of neighbors
    neighbors = knn(pSets[:50], fTypes, pSets[p0], kk)  # start from p0, collect all 6 nearest restaurant
    print('\n\n######################## Non Dominated #################################')
    for nd in neighbors:
        print('Non Dominated:', nd)

tEnd = time.time()
print("\nTotal time: ", tEnd - tStart, "seconds")
