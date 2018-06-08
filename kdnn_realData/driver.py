"""
Created on 6/7/18

@author: Jingchao Yang
"""
############## data source ##############
# from kdnn_realData import dataCollector_Yelp
# from kdnn_realData import dataCollector_News
# from kdnn_realData import dataCollector_Pubmed
import dataGenerator

############## algorithms ##############
from kdnn_realData import KDNN_Entropy_Greedy
from kdnn_realData import KDNN_Entropy_Hybrid
from kdnn_realData import KDNN_Entropy_Optimal
from kdnn_realData import KDNN_Union_Greedy
from kdnn_realData import KDNN_Union_Optimal

import time
import random

if __name__ == "__main__":
    tStart = time.time()

    kk = 6  # Num of neighbors
    datasetRange = 20

    loops = 100

    # ############## Yelp #################
    # pSets = dataCollector_Yelp.allPoints
    # fTypes = dataCollector_Yelp.allCategories
    # userAddr = [(35.04728681, -80.99055881)]  # input user location

    # ############## News #################
    # pSets = dataCollector_News.allPoints
    # fTypes = dataCollector_News.allCategories
    # # userAddr = [(-12.97221841, -38.50141361)]  # input user location

    # ############## Publication #################
    # pSets = dataCollector_Pubmed.allPoints
    # fTypes = dataCollector_Pubmed.allCategories
    # userAddr = [(52.15714851, 4.4852091)]  # input user location

    ############## Synthetic Data #################
    x, y = 0, 0
    userAddr = [(x, y)]  # input user location
    numRecords = 100
    searchRange = 500
    cateNum = 10
    cateRange = 200
    pSets, fTypes = dataGenerator.randomData(numRecords, searchRange, x, y, cateNum, cateRange)

    for i in range(loops):
        id = random.randint(0, datasetRange-1)
        getLoc = pSets[id]
        print('\n@@@@@@@@@@@@ User ID/Location', id, getLoc, '@@@@@@@@@@@@')

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

        # select an algorithm for kdnn
        neighbors = KDNN_Entropy_Greedy.knn(pSets[:datasetRange], fTypes, id,
                                            kk, ftWW)  # start from p0, collect all 6 nearest restaurant
        print('\n\n######################## Non Dominated #################################')
        for nd in neighbors:
            print('(Nbor, Dist, Div, RT): ', nd)

tEnd = time.time()
print("\nTotal time: ", tEnd - tStart, "seconds")
