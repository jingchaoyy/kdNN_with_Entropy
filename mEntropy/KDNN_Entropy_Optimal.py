"""
Created on 7/31/18

@author: Jingchao Yang
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import entropy
import time
import itertools
from mEntropy import assignWeights


def knn(pS, fTs, pid, k, wFTs):
    """
    Entropy enabled knn Algorithm will compute the diversity/ entropy each time when a new neighbor added choose k out
    of total n neighbors found, and calculate to have the max entropy sets. 
    In a way, it provides all non-dominated sets
    
    :param pS: all latlons
    :param fTs: all fTs associated with pS
    :param pid: user location, represented by restaurant ID
    :param k: number of nbors
    :param wFTs: set combine fTs with weight
    :return: Best subset with k nbors from found set
    """

    X = np.array(pS)
    nonDominated = []
    targetPNbrs = ''
    for i in range(len(pS)):
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
            knnT = assignFT(fTs, tnList)
            print('Original', knnT)

            # when more than k neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(tnList) >= k:

                div_local = assignWeights.assignWeights(knnT, wFTs, tnList)
                print('@@@@@', div_local)

                subs = findsubsets(tnList, k)  # get all the combinations (choose k out of n)
                # neighborsAfter = checkNeighbor(fTs, subs)

                runTStart = time.time()
                resultNbor = checkNeighbor(fTs, subs, wFTs)
                runTEnd = time.time()
                neighborsAfter = resultNbor[0]  # set of neighbors
                divAfter = resultNbor[1]  # entropy of the neighbor set
                runT = runTEnd - runTStart  # get the runtime

                nbor_localDiv = []  # collect local entropy for the selected nbors
                for nbor in neighborsAfter:
                    for dl in div_local:
                        if nbor in dl:
                            nbor_localDiv.append(dl[0])
                bestDiv_loc = sum(nbor_localDiv)

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

                nonDominated.append([maxDist, bestDiv_loc, runT])

    print('\n\n######################## Original Vs. Final Results #################################')
    print('Original Neighbors:', tnList)
    print('Original Food Type Rank:', assignFT(fTs, tnList))

    return nonDominated


""" Function defined to output all subsets
(Choose k out of n)
"""


def findsubsets(S, k):
    """
    Function defined to output all subsets (Choose k out of n)
    
    :param S: S choose k
    :param k: S choose k
    :return: all possible subsets
    """
    return set(itertools.combinations(S, k))


def assignFT(fTs, nbors):
    """
    Assign food types to neighbors
    
    :param fTs: foot types list, matched index with nbors
    :param nbors: list of nbors
    :return: list of food types 
    """
    neighborTypes = []
    # assign each neighbor with their color type
    for n in nbors:
        neighborTypes.append(
            fTs[n])
    return neighborTypes


def checkNeighbor(fTs, subsets, wFTs):
    """
    Function defined to check if a switch required between tha last two neighbor based on the entropy calculated
    
    :param fTs: food type list
    :param subsets: subseted neighbor list
    :param wFTs: set combine fTs with weight
    :return: selected neighbor set, with its local entropy
    """
    subsetFTList, subsetList = [], []
    for ss in subsets:
        # assign food type to all the neighbors first
        subsetFTList.append(assignFT(fTs, ss))
        subsetList.append(ss)
        # calculating the diversity without the newly added neighbor
    divList = []
    for i in subsetFTList:  # in each subset
        sets = []
        for j in i:  # in each restaurant
            for k in j:  # getting each category
                sets.append(k)
        diversity = entropy.calcShannonEnt(sets, wFTs)
        divList.append(diversity)

    # looking for the max entropy
    bestDiv = max(divList)
    # getting index of the best neighbor sets, even if there are more then one record matches the max entropy
    # the reason is that the entropy value added to the div list with a sequence that minimize the distance
    # meaning the one in the front has lower total distance
    bestIndex = divList.index(bestDiv)

    nbors = list(subsetList[bestIndex])
    # print(nbors)
    return nbors, bestDiv
