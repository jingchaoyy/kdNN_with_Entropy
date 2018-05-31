"""
Created on 5/18/18

@author: Jingchao Yang
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kdnn_realData import yelpDataCollector
import entropy
import time
import itertools

tStart = time.time()

""" Entropy enabled knn
Algorithm will compute the diversity/ entropy each time when a new neighbor added
choose k out of total n neighbors found, and calculate to have the max entropy sets. 
In a way, it provides all non-dominated sets
"""


def knn(pS, fTs, pLatLon, k):
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
            targetDistances = distances[pLocation]

            tnList = list(targetPNbrs)
            tdList = list(targetDistances)

            # when more than k neighbors found, check if a switch of the last two can improve the diversity,
            # and return the adjusted neighbor list
            if len(targetPNbrs) > k:
                subs = findsubsets(targetPNbrs, k)  # get all the combinations (choose k out of n)
                neighborsAfter = checkNeighbor(fTs, subs)

                distanceAfter = []
                for na in neighborsAfter:
                    if na in tnList:
                        ind = tnList.index(na)
                        distanceAfter.append(tdList[ind])
                maxDist = max(distanceAfter)

                print('Adjusted', assignFT(fTs, neighborsAfter))
                print('nondominated neighbor set:', (neighborsAfter[:k], maxDist))

                if nonDominated[-1] != (
                        neighborsAfter[:k], maxDist):  # see if the last added nonDominated sets is tha same
                    #  as the latest one, if the same, ignore the latest one
                    nonDominated.append((neighborsAfter[:k], maxDist))

            else:  # when less than minimum required neighbors found, add to the neighbor list directly
                print('nondominated neighbor set:', targetPNbrs)
                if len(targetPNbrs) == k:  # add the first find knn set to the nonDominated list
                    maxDisttd = max(tdList)
                    nonDominated.append((tnList, maxDisttd))

    print('\n\n######################## Original Vs. Final Results #################################')
    print('Original Neighbors:', targetPNbrs)
    print('Original Food Type Rank:', assignFT(fTs, targetPNbrs))
    print('Final Neighbors:', targetPNbrs)
    print('Final Food Type Rank:', assignFT(fTs, targetPNbrs))

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


def checkNeighbor(fTs, subsets):
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
        diversity = entropy.calcShannonEnt(sets)
        divList.append(diversity)

    # looking for the max entropy
    bestDiv = max(divList)
    # getting index of the best neighbor sets, even if there are more then one record matches the max entropy
    # the reason is that the entropy value added to the div list with a sequence that minimize the distance
    # meaning the one in the front has lower total distance
    bestIndex = divList.index(bestDiv)

    nbors = list(subsetList[bestIndex])
    # print(nbors)
    return nbors


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

    k = 6  # Num of neighbors
    neighbors = knn(pSets[:50], fTypes, pSets[p0], k)  # start from p0, collect all 6 nearest restaurant
    print('\n\n######################## Non Dominated #################################')
    for nd in neighbors:
        print('Non Dominated:', nd)

tEnd = time.time()
print("\nTotal time: ", tEnd - tStart, "seconds")
