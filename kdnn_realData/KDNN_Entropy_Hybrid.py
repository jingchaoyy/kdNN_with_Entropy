"""
Created on 5/27/18

@author: Jingchao Yang
"""

import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

import entropy
from toolBox import entropyGetWeight

""" Entropy enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
Using all categories found within neighbors to calculate entropy weight for each category, 
and use the weighted entropy to calculated the final total entropy for each neighbor,
choose k that ranked as the highest
"""


def knn(pS, fTs, pid, k, wFTs):
    # pS: all latlons
    # fTs: all fTs associated with pS
    # pid: user location, represented by restaurant ID
    # k: number of nbors
    # wFTs: set combine fTs with weight

    X = np.array(pS)
    nonDominated = []
    for i in range(len(pS)):
        if i > 1:  # at least one nearest neighbor (i.e. itself)
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points)
            distances, neighbors = nbrs.kneighbors(X)
            # retrieving neighbors for target point
            targetPNbrs = neighbors[pid]
            targetDistances = distances[pid]

            tnList = list(targetPNbrs)  # starting from the second one, first one is the user itself
            print("\nOriginal NID", tnList)
            print('Original', assignFT(fTs, tnList))
            tdList = list(targetDistances)

            maxDist = max(tdList)  # max distance within all found neighbors (search range)

            # when more than k neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(tnList) >= k:

                runTStart = time.time()
                resultNbor = checkNeighbor(fTs, tnList, k, wFTs)
                runTEnd = time.time()
                neighborsAfter = resultNbor[0]  # set of neighbors
                # divAfter = resultNbor[1]  # entropy of the neighbor set
                runT = runTEnd - runTStart  # get the runtime

                print('Adjusted NID', neighborsAfter)
                catAfter = assignFT(fTs, neighborsAfter)
                print('Adjusted', catAfter)

                catList = []
                for a in catAfter:
                    for b in a:
                        catList.append(b)

                divAfter = entropy.calcShannonEnt(catList, wFTs)

                # print('nondominated neighbor set:', bestDiv)

                distanceAfter = []
                for na in neighborsAfter:
                    if na in tnList:
                        ind = tnList.index(na)
                        distanceAfter.append(tdList[ind])
                # maxDist = max(distanceAfter)

                # if len(tnList) == k:
                #     nonDominated.append((neighborsAfter, maxDist, divAfter, runT))
                #
                # elif set(nonDominated[-1][0]) != set(
                #         neighborsAfter):  # see if the last added nonDominated sets is tha same
                #     # as the latest one, if the same, ignore the latest one
                #     nonDominated.append((neighborsAfter, maxDist, divAfter, runT))

                nonDominated.append([maxDist, divAfter, runT])

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


def getKey(item):
    return item[0]


""" Function defined to gather weight info for each category during 
entropy calculation, rank each restaurant based on the total weighted
entropy, and output k restaurants with highest entropy value
"""


def checkNeighbor(fTs, nbors, kk, wFTs):
    # assign food type to all the neighbors first
    knnT = assignFT(fTs, nbors)

    ftList = []
    for x in knnT:
        for y in x:
            ftList.append(y)

    weights = entropyGetWeight.calcShannonEnt(ftList, wFTs)[1]

    div = []  # list for recording total entropy of each neighbor, and related neighbor
    for i in range(len(knnT)):
        iEntropy = 0  # for recording the entropy of neighbor i
        for j in knnT[i]:
            for k in weights:
                if j == k[0]:  # assign weight of categories to each category in that neighbor
                    iEntropy += k[1]
        div.append((iEntropy, nbors[i]))
    print("divOriginal", div)

    # div.sort(reverse=True)
    divSort = sorted(div, key=getKey,
                     reverse=True)  # reverse sort list based on only the first element (entropy) in each tuple,
    # from largest weighted div to smallest

    print("divAdjusted", divSort)

    knbors, kdiv = [], []
    for z in divSort:
        knbors.append(z[1])  # collect neighbors
        kdiv.append(z[0])
    knbors = knbors[:kk]  # get the first 6, as kk == 6
    kdiv = kdiv[:kk]

    bestDiv = sum(kdiv)

    neighborList = []  # put the selected neighbors in its original order (distance based)
    for a in nbors:
        for b in knbors:
            if a == b:
                neighborList.append(a)

    return neighborList, bestDiv
