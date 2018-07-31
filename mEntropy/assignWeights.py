"""
Created on 7/31/18

@author: YJccccc
"""
from toolBox import entropyGetWeight


def assignWeights(knnT, wFTs, nbors):
    """
    
    :param knnT: neighbor associated food type set
    :param wFTs: food type list
    :param nbors: neighbor list
    :return: local diversity for each neighbor 
    """
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
    return div
