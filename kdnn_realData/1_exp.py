"""
Created on 6/9/18

@author: Jingchao Yang
"""
############## data source ##############
# from kdnn_realData import dataCollector_Yelp
# from kdnn_realData import dataCollector_News
from kdnn_realData import dataCollector_Pubmed
# import dataGenerator

############## algorithms ##############
from kdnn_realData import KDNN_Entropy_Greedy
from kdnn_realData import KDNN_Entropy_Hybrid
from kdnn_realData import KDNN_Entropy_Optimal
from kdnn_realData import KDNN_Union_Greedy
from kdnn_realData import KDNN_Union_Optimal

import time
import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tStart = time.time()

    kk = 6  # Num of neighbors
    datasetRange = 20
    loops = 5

    # ############## Yelp #################
    # pSets = dataCollector_Yelp.allPoints
    # fTypes = dataCollector_Yelp.allCategories
    # userAddr = [(35.04728681, -80.99055881)]  # input user location
    #
    # ############## News #################
    # pSets = dataCollector_News.allPoints
    # fTypes = dataCollector_News.allCategories
    # # userAddr = [(-12.97221841, -38.50141361)]  # input user location

    ############## Publication #################
    pSets = dataCollector_Pubmed.allPoints
    fTypes = dataCollector_Pubmed.allCategories
    userAddr = [(52.15714851, 4.4852091)]  # input user location

    # ############# Synthetic Data #################
    # x, y = 0, 0
    # numRecords = 100
    # searchRange = 500
    # cateNum = 10
    # cateRange = 200
    # pSets, fTypes = dataGenerator.randomData(numRecords, searchRange, x, y, cateNum, cateRange)

    algorithms = [KDNN_Entropy_Greedy, KDNN_Entropy_Hybrid, KDNN_Entropy_Optimal]
    colors = ['r', 'g', 'b']
    labels = ["Entropy_Greedy", "Entropy_Hybrid", "Entropy_Optimal"]

    preferences = []
    users = range(0, datasetRange)  # define users
    userList = random.sample(users,loops)  # random pick users for loops number of times

    for x in range(datasetRange):  # generating preferences
        allFt, preWeight = [], []
        for ftSet in fTypes:  # get all fts (with duplicates)
            for ft in ftSet:
                allFt.append(ft)
        allFt = KDNN_Union_Greedy.Remove(allFt)  # ft without duplicate
        for j in range(len(allFt)):
            preWeight.append(random.uniform(0, 1))

        ftWW = []  # bound ft with weight
        for k in range(len(allFt)):
            ftWW.append((allFt[k], preWeight[k]))

        preferences.append(ftWW)

    fig, ax = plt.subplots()
    for i in range(len(algorithms)):
        resultPool = []
        # if i == 2:
        #     users = range(0, 20)
        #     datasetRange = 20

        for j in range(loops):
            # select an algorithm for kdnn
            neighbors = algorithms[i].knn(pSets[:datasetRange], fTypes, userList[j], kk, preferences[userList[j]])
            resultPool.append(neighbors)

        print('\n\n######################## Non Dominated #################################')
        print('       Dist         Diversity       Runtime')
        resultPool = np.array(resultPool)
        resultPool = resultPool.sum(axis=0)  # sum by columns
        avg = resultPool / loops  # all get average
        print(avg)

        X = avg[:, 0]  # Dist
        Y = avg[:, 1]  # Diversity
        Z = avg[:, 2]  # Runtime

        ax.plot(X, Y, colors[i], label=labels[i])

    legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
    plt.title("Correlation Between Distance and Diversity")
    plt.xlabel("Distance")
    plt.ylabel("Diversity")
    plt.show()

    tEnd = time.time()
    print("\nTotal time: ", tEnd - tStart, "seconds")
