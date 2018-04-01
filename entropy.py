from math import log

def calcShannonEnt(dataSet):
    countDataSet = len(dataSet)
    typeCounts={}
    for restaurant in dataSet:
        currentLabel=restaurant[-1]
        if currentLabel not in typeCounts.keys():
            typeCounts[currentLabel] = 0
        typeCounts[currentLabel] += 1

    print (typeCounts)

    shannonEnt = 0.0

    for type in typeCounts:
        prob = float(typeCounts[type])/countDataSet
        # print prob
        shannonEnt -= prob * log(prob,2)

    return shannonEnt

dataSet1 = [['Chinese'],['Chinese']]
dataSet2 = [['Chinese'], ['Japanese']]
dataSet3 = [['Chinese'], ['Japanese'], ['American']]

############ entropy for base set ##############
dataSet4 = [['Chinese'], ['Japanese'], ['Italian'], ['American']]

############ entropy after enlarge set data ##############
dataSet5 = [['Chinese'], ['Chinese'], ['Japanese'], ['Italian'], ['American']]
dataSet6 = [['Chinese'], ['Japanese'], ['Japanese'], ['Italian'], ['American']]
dataSet7 = [['Chinese'], ['Japanese'], ['Italian'], ['Italian'], ['American']]
dataSet8 = [['Chinese'], ['Japanese'], ['Italian'], ['American'], ['American']]
# dataSet9 = [['Chinese'], ['Japanese'], ['Italian'], ['American'],['Chines'], ['Japanes'], ['Italia'], ['America']]

data = [dataSet1,dataSet2,dataSet3,dataSet4,dataSet5,dataSet6,dataSet7,dataSet8]

for d in data:
    print (calcShannonEnt(d))

# due to Chinese food are more similar to Japanese food than Italian to American
# print '############ with weight ##############'
# print calcShannonEnt(dataSet5) - 0.2
# print calcShannonEnt(dataSet6) - 0.2
# print calcShannonEnt(dataSet7) - 0.1
# print calcShannonEnt(dataSet8) - 0.1

