"""
Created on 6/9/18

@author: Jingchao Yang
"""
############## data source ##############
import dataGenerator

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
import csv

if __name__ == "__main__":
   kk = 6  # Num of neighbors
   datasetRange = 50
   loops = 10

   ############# Synthetic Data #################
   x, y = 0, 0
   numRecords = 500
   searchRange = 500
   cateNum = [3, 5, 10]  # 3, 5, 10
   cateRange = 200
   colors = ['r', 'g', 'b']

   fig, ax = plt.subplots()

   for i in range(len(cateNum)):
       pSets, fTypes = dataGenerator.randomData(numRecords, searchRange, x, y, cateNum[i], cateRange)
       resultPool = []
       runtimes = []
       for n in range(datasetRange):
           if n >1:
               tStart = time.time()
               for j in range(loops):
                   id = random.randint(0, n - 1)
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
                   neighbors = KDNN_Entropy_Greedy.knn(pSets[:n], fTypes, id,
                                                 kk, ftWW)  # start from p0, collect all 6 nearest restaurant


               tEnd = time.time()
               runtimes.append(tEnd - tStart)

       with open('/Users/YJccccc/kdNN_with_Entropy/kdnn_realData/results/exp3/Entropy' + str(cateNum[
           i]) + '.csv', 'w',
                 newline='') as csvfile:
           spamwriter = csv.writer(csvfile, delimiter=' ',
                                   quotechar='|', quoting=csv.QUOTE_MINIMAL)
           print(runtimes)
           for r in range(len(runtimes)):
               spamwriter.writerow([r + 2, runtimes[r]])
   #     ax.plot(range(2,datasetRange), runtimes, colors[i], label=cateNum[i])
   #
   # legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
   # plt.title("KDNN_Entropy_Greedy: Correlation Between Neighbor Size and Runtime")
   # plt.xlabel("Neighbor Size")
   # plt.ylabel("Runtime")
   # plt.show()

