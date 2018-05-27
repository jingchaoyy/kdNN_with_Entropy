"""
Created on 5/27/18

@author: Jingchao Yang
"""

from math import log


def calcShannonEnt(dataSet):
    countDataSet = len(dataSet)
    typeCounts = {}
    for restaurant in dataSet:
        currentLabel = restaurant
        if currentLabel not in typeCounts.keys():
            typeCounts[currentLabel] = 0
        typeCounts[currentLabel] += 1

    print(typeCounts)
    shannonEnt = 0.0

    wList = []
    for type in typeCounts:
        prob = float(typeCounts[type]) / countDataSet
        weight = log(prob, 2)
        wList.append((type, abs(weight)))
        shannonEnt -= log(prob, 2)

    return shannonEnt, wList


# dataSet1 = ['Chinese', 'Chinese']
# dataSet2 = ['Chinese', 'Japanese']
# dataSet3 = ['Chinese', 'Japanese', 'American']
#
# ############ entropy for base set ##############
# dataSet4 = ['Chinese', 'Japanese', 'Italian', 'American']
#
# ############ entropy after enlarge set data ##############
# dataSet5 = ['Chinese', 'Chinese', 'Japanese', 'Italian', 'American']
# dataSet6 = ['Chinese', 'Chinese', 'Japanese', 'Japanese', 'American']
# dataSet7 = ['Chinese', 'Chinese', 'Chinese', 'Japanese', 'Japanese']
# dataSet8 = ['Chinese', 'Chinese', 'Chinese', 'Chinese', 'Japanese']
# # dataSet9 = [['Chinese'], ['Japanese'], ['Italian'], ['American'],['Chinese'], ['Japanes'], ['Italia'], ['America']]
#
# data = [dataSet1, dataSet2, dataSet3, dataSet4, dataSet5, dataSet6, dataSet7, dataSet8]
#
# for d in data:
#     print(calcShannonEnt(d))
