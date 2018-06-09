"""
Created on 5/31/18

@author: Jingchao Yang
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time
import itertools
from kdnn_realData import KDNN_Union_Greedy

""" Union enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
choose k out of k+1 neighbors found, and calculate to have a set with the max category types. 
In a way, it provides all non-dominated sets
"""


# same as KDNN_Greedy
def knn(pS, fTs, pid, k, wFTs):
    # pS: all latlons
    # fTs: all fTs associated with pS
    # pid: user location, represented by restaurant ID
    # k: number of nbors
    # wFTs: set combine fTs with weight

    X = np.array(pS)
    nonDominated = []
    targetPNbrs = ''
    for i in range(len(pS) + 1):
        if i > 0:  # at least one nearest neighbor (i.e. itself)
            nbrs = NearestNeighbors(n_neighbors=i, algorithm='ball_tree').fit(X)
            # return distances and ranked neighbors (presented as point location in array points)
            distances, neighbors = nbrs.kneighbors(X)
            # retrieving neighbors for target point
            targetPNbrs = neighbors[pid]
            targetDistances = distances[pid]

            tnList = list(targetPNbrs)
            tdList = list(targetDistances)

            maxDist = max(tdList)  # max distance within all found neighbors (search range)

            print('\nOriginal NID', tnList)
            print('Original', assignFT(fTs, tnList))

            # when more than k neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(tnList) >= k:
                subs = findsubsets(tnList, k)  # get all the combinations (choose k out of n)
                # neighborsAfter = checkNeighbor(fTs, subs)

                runTStart = time.time()
                resultNbor = checkNeighbor(fTs, subs, wFTs)
                runTEnd = time.time()
                neighborsAfter = resultNbor[0]  # set of neighbors
                divAfter = resultNbor[1]  # entropy of the neighbor set
                runT = runTEnd - runTStart  # get the runtime

                distanceAfter = []
                for na in neighborsAfter:
                    if na in tnList:
                        ind = tnList.index(na)
                        distanceAfter.append(tdList[ind])
                # maxDist = max(distanceAfter)  # max distance within selected neighbor

                print('Adjusted NID', neighborsAfter)
                print('Adjusted', assignFT(fTs, neighborsAfter))
                print('nondominated neighbor set:', (neighborsAfter[:k], maxDist))

                # if len(tnList) == k:
                #     nonDominated.append((neighborsAfter, maxDist, divAfter, runT))
                #
                # elif set(nonDominated[-1][0]) != set(
                #         neighborsAfter):  # see if the last added nonDominated sets is tha same
                #     # as the latest one, if the same, ignore the latest one
                #     nonDominated.append((neighborsAfter, maxDist, divAfter, runT))

                nonDominated.append([maxDist, divAfter, runT])

    print('\n\n######################## Original Vs. Final Results #################################')
    print('Original Neighbors:', tnList)
    print('Original Food Type Rank:', assignFT(fTs, tnList))

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
        neighborTypes.append(
            fTs[n])  # n - 1 due to the sequence all push back by 1 after adding user location to the locationlist
    return neighborTypes


""" Function defined to check if a switch required between tha last two neighbor
based on the entropy calculated
"""


def checkNeighbor(fTs, subsets, wFTs):
    subsetFTList, subsetList = [], []
    for ss in subsets:
        # assign food type to all the neighbors first
        subsetFTList.append(assignFT(fTs, ss))
        subsetList.append(ss)
        # calculating the diversity without the newly added neighbor
    div = []
    for i in subsetFTList:  # in each subset
        sets = []  # collecting all categories from all restaurants
        for j in i:
            for k in j:
                sets.append(k)

        sets = KDNN_Union_Greedy.Remove(sets)
        diversity = KDNN_Union_Greedy.assignWeight(sets, wFTs)
        div.append(diversity)

    # looking for the max entropy
    bestDiv = max(div)
    # getting index of the best neighbor sets, even if there are more then one record matches the max entropy
    # the reason is that the entropy value added to the div list with a sequence that minimize the distance
    # meaning the one in the front has lower total distance
    bestIndex = div.index(bestDiv)

    nbors = list(subsetList[bestIndex])
    # print(nbors)
    return nbors, bestDiv
