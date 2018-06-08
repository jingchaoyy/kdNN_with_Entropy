"""
Created on 5/14/18

@author: Jingchao Yang
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import entropy
import time
import itertools

""" Entropy enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
choose k out of k+1 neighbors found, and calculate to have the max entropy sets. 
In a way, it provides all non-dominated sets
"""


def knn(pS, fTs, pid, k, wFTs):
    # pS: all latlons
    # fTs: all fTs associated with pS
    # pid: user location, represented by restaurant ID
    # k: number of nbors
    # wFTs: set combine fTs with weight

    X = np.array(pS)
    neighborsAfter = []  # Storing neighbors after each time getting neighbor check and switched

    # looking for knn one by one
    nonDominated = []
    lastNbors = 0  # record the original nbor set for last iteration
    for i in range(len(pS) + 1):
        if i > 0:  # at least two nearest neighbor (i.e. [0] --> user)
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points)
            distances, neighbors = nbrs.kneighbors(X)
            # retrieving neighbors for user point
            targetPNbrs = neighbors[pid]
            targetDistances = distances[pid]

            tnList = list(targetPNbrs)  # starting from the second one, first one is the user itself
            tdList = list(targetDistances)

            # adding the latest neighbor to the adjusted neighbor list, total will always be k+1
            if lastNbors != 0:
                neighborsAfter.append(list(set(tnList) - set(lastNbors))[0])
            else:
                neighborsAfter.append(tnList[0])
            print('\nOriginal', neighborsAfter)
            print('Original', assignFT(fTs, neighborsAfter))

            # when more than k neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(neighborsAfter) > k:

                runTStart = time.time()
                resultNbor = checkNeighbor(fTs, neighborsAfter, wFTs)
                runTEnd = time.time()
                neighborsAfter = resultNbor[0]  # set of neighbors
                divAfter = resultNbor[1]  # entropy of the neighbor set
                runT = runTEnd - runTStart  # get the runtime

                distanceAfter = []
                for na in neighborsAfter:
                    if na in tnList:
                        ind = tnList.index(na)
                        distanceAfter.append(tdList[ind])
                maxDist = max(distanceAfter)

                print('Adjusted', assignFT(fTs, neighborsAfter))
                print('nondominated neighbor set:', neighborsAfter)
                if set(nonDominated[-1][0]) != set(
                        neighborsAfter):  # see if the last added nonDominated sets is tha same
                    # as the latest one, if the same, ignore the latest one
                    nonDominated.append((neighborsAfter[:k], maxDist, divAfter, runT))

            else:  # when less than minimum required neighbors found, add to the neighbor list directly
                neighborsAfter = neighborsAfter
                print('nondominated neighbor set:', neighborsAfter)
                if len(neighborsAfter) == k:  # add the first find knn set to the nonDominated list
                    maxDisttd = max(tdList)
                    runTStart1 = time.time()
                    atts = assignFT(fTs, tnList)
                    attSets = []
                    for att in atts:
                        for a in att:
                            attSets.append(a)
                    diversity = entropy.calcShannonEnt(attSets, wFTs)
                    runTEnd1 = time.time()
                    runT1 = runTEnd1 - runTStart1  # get the runtime
                    nonDominated.append((neighborsAfter[:k], maxDisttd, diversity, runT1))

            lastNbors = tnList

    print('\n\n######################## Original Vs. Final Results #################################')
    print('Original Neighbors:', tnList)
    print('Original Food Type Rank:', assignFT(fTs, tnList))
    print('Final Neighbors:', neighborsAfter)
    print('Final Food Type Rank:', assignFT(fTs, neighborsAfter))

    return nonDominated


""" Function defined to output all subsets
(Choose k out of n)
"""


def findsubsets(S, k):
    return set(itertools.combinations(S, k))


""" Function defined to output all neighbors with 
assigned food types for input point latlon
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


def checkNeighbor(fTs, nbors, wFTs):
    # assign food type to all the neighbors first
    knnT = assignFT(fTs, nbors)
    # calculating the diversity without the newly added neighbor

    div = []
    for i in range(len(nbors)):
        # computing entropy backwards, and collecting entropy values
        sets = []  # collecting all categories from all restaurants
        for j in knnT[:len(knnT) - i - 1]:
            for k in j:
                sets.append(k)
        for z in knnT[len(knnT) - i:len(knnT)]:
            for y in z:
                sets.append(y)

        diversity = entropy.calcShannonEnt(sets, wFTs)
        div.append(diversity)
        print(sets)
        print(diversity)
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
    # print(nbors)
    return nbors, bestDiv
