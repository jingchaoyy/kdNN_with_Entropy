"""
Created on 6/9/18

@author: Jingchao Yang
"""
############## data source ##############
from kdnn_realData import dataCollector_Yelp
from kdnn_realData import dataCollector_News
from kdnn_realData import dataCollector_Pubmed
import dataGenerator

############## algorithms ##############
from kdnn_realData import KDNN_Entropy_Greedy
from kdnn_realData import KDNN_Entropy_Hybrid
from kdnn_realData import KDNN_Entropy_Optimal
from kdnn_realData import KDNN_Union_Greedy
from kdnn_realData import KDNN_Union_Optimal
import csv

import time
import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    kk = 6  # Num of neighbors
    datasetRange = 10
    loops = 1

    ############## Yelp #################
    pSets_yelp = dataCollector_Yelp.allPoints
    fTypes_yelp = dataCollector_Yelp.allCategories

    ############## News #################
    pSets_news = dataCollector_News.allPoints
    fTypes_news = dataCollector_News.allCategories

    # ############## Publication #################
    pSets_pub = dataCollector_Pubmed.allPoints
    fTypes_pub = dataCollector_Pubmed.allCategories

    ############# Synthetic Data #################
    x, y = 0, 0
    numRecords = 500
    searchRange = 500
    cateNum = 3  # 3, 5, 10
    cateRange = 200
    pSets_syn, fTypes_syn = dataGenerator.randomData(numRecords, searchRange, x, y, cateNum, cateRange)

    datasets = [pSets_yelp]
    dataName = ['Yelp']
    datasetsType = [fTypes_yelp, fTypes_news, fTypes_pub, fTypes_syn]

    algorithms = [KDNN_Entropy_Greedy, KDNN_Entropy_Hybrid, KDNN_Entropy_Optimal]
    colors = ['r', 'g', 'b']
    labels = ["Entropy_Greedy", "Entropy_Hybrid", "Entropy_Optimal"]

    users = range(datasetRange)

    for ds in range(len(datasets)):
        # print(k)
        preferences = []
        # fig, ax = plt.subplots()
        for x in range(datasetRange):  # generating preferences
            allFt, preWeight = [], []
            for ftSet in datasetsType[ds]:  # get all fts (with duplicates)
                for ft in ftSet:
                    allFt.append(ft)
            allFt = KDNN_Union_Greedy.Remove(allFt)  # ft without duplicate
            for j in range(len(allFt)):
                preWeight.append(random.uniform(0, 1))

            ftWW = []  # bound ft with weight
            for k in range(len(allFt)):
                ftWW.append((allFt[k], preWeight[k]))

            preferences.append(ftWW)

        for i in range(len(algorithms)):
            if i == 2:
                users = range(0, 20)
                datasetRange = 20

            resultPool = []
            runtimes = []

            for n in range(datasetRange):
                if n > 1:
                    tStart = time.time()

                    for j in range(loops):
                        users = users[:n]
                        user = random.choice(users)

                        # select an algorithm for kdnn
                        neighbors = algorithms[i].knn(datasets[ds][:n], datasetsType[ds], user, kk, preferences[user])

                    tEnd = time.time()
                    runtimes.append(tEnd - tStart)


            with open('/Users/YJccccc/kdNN_with_Entropy/kdnn_realData/results/exp2/' + dataName[ds] + labels[
                i] + '.csv', 'w',
                      newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                print(runtimes)
                for r in range(len(runtimes)):
                    spamwriter.writerow([r+2, runtimes[r]])
    #         ax.plot(range(2, datasetRange), runtimes, colors[i], label=labels[i])
    #
    # legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
    # plt.title("Correlation Between Neighbor Size and Runtime")
    # plt.xlabel("Neighbor Size")
    # plt.ylabel("Runtime")
    # plt.show()
