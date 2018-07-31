"""
Created on 6/9/18

@author: Jingchao Yang
"""
############## data source ##############
# from kdnn_realData import dataCollector_Yelp
# from kdnn_realData import dataCollector_News
# from kdnn_realData import dataCollector_Pubmed
import dataGenerator

############## algorithms ##############
from mEntropy import KDNN_Entropy_Greedy
from mEntropy import KDNN_Entropy_Hybrid
from mEntropy import KDNN_Entropy_Optimal
from kdnn_realData import KDNN_Union_Greedy
from kdnn_realData import KDNN_Union_Optimal

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

if __name__ == "__main__":
    tStart = time.time()

    kk = 6  # Num of neighbors
    datasetRange = 30
    loops = 10

    # ############## Yelp #################
    # pSets = dataCollector_Yelp.allPoints
    # fTypes = dataCollector_Yelp.allCategories
    # userAddr = [(35.04728681, -80.99055881)]  # input user location
    #
    # ############## News #################
    # pSets = dataCollector_News.allPoints
    # fTypes = dataCollector_News.allCategories
    # # userAddr = [(-12.97221841, -38.50141361)]  # input user location

    # ############## Publication #################
    # pSets = dataCollector_Pubmed.allPoints
    # fTypes = dataCollector_Pubmed.allCategories
    # userAddr = [(52.15714851, 4.4852091)]  # input user location

    ############# Synthetic Data #################
    x, y = 0, 0
    numRecords = 100
    searchRange = 500
    cateNum = 5
    cateRange = 10
    pSets, fTypes = dataGenerator.randomData(numRecords, searchRange, x, y, cateNum, cateRange)

    algorithms = [KDNN_Entropy_Greedy, KDNN_Entropy_Hybrid, KDNN_Entropy_Optimal]
    colors = ['r', 'g', 'b']
    labels = ["Entropy_Greedy", "Entropy_Hybrid", "Entropy_Optimal"]

    preferences = []
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

    ############### Test run and csv write
    # fig, ax = plt.subplots()
    for i in range(len(algorithms)):
        resultPool = []
        # if i == 2:
        #     loops = 10

        for j in range(loops):
            # select an algorithm for kdnn
            user = j % datasetRange
            neighbors = algorithms[i].knn(pSets[:datasetRange], fTypes, user, kk, preferences[user])
            resultPool.append(neighbors)

        print('\n\n######################## Non Dominated #################################')
        print('       Dist         Diversity       Runtime')
        resultPool = np.array(resultPool)
        resultPool = resultPool.sum(axis=0)  # sum by columns
        avg = resultPool / loops  # all get average

        X = avg[:, 0]  # Dist
        Y = avg[:, 1]  # Diversity
        Z = avg[:, 2]  # Runtime


        with open('/Users/YJccccc/kdNN_with_Entropy/Possible_Result/mEntropy_exp1/'+labels[i]+'.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for r in avg:
                spamwriter.writerow(r)

    tEnd = time.time()
    print("\nTotal time: ", tEnd - tStart, "seconds")
