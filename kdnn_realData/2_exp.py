"""
Created on 6/9/18

@author: Jingchao Yang
"""
import csv
import random
import time

############## algorithms ##############
from kdnn_realData import KDNN_Entropy_Greedy
from kdnn_realData import KDNN_Entropy_Hybrid
from kdnn_realData import KDNN_Entropy_Optimal
from kdnn_realData import KDNN_Union_Greedy
from kdnn_realData import KDNN_Union_Optimal
from toolBox import dataCollector_News
############## data source ##############
from toolBox import dataCollector_Yelp
from toolBox import dataCollector_Pubmed

if __name__ == "__main__":
    kk = 6  # Num of neighbors
    datasetRange = 50
    loops = 10

    ############## Yelp #################
    pSets_yelp = dataCollector_Yelp.allPoints
    fTypes_yelp = dataCollector_Yelp.allCategories

    ############## News #################
    pSets_news = dataCollector_News.allPoints
    fTypes_news = dataCollector_News.allCategories

    # ############## Publication #################
    pSets_pub = dataCollector_Pubmed.allPoints
    fTypes_pub = dataCollector_Pubmed.allCategories

    datasets = [pSets_yelp, pSets_news, pSets_pub]
    dataName = ['Yelp', 'News', 'Pub']
    datasetsType = [fTypes_yelp, fTypes_news, fTypes_pub]

    algorithms = [KDNN_Entropy_Greedy, KDNN_Entropy_Hybrid, KDNN_Entropy_Optimal, KDNN_Union_Greedy, KDNN_Union_Optimal]
    # colors = ['r', 'g', 'b']
    labels = ["Entropy_Greedy", "Entropy_Hybrid", "Entropy_Optimal", "Union_Greedy", "Union_Optimal"]

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
            if i == 2 or i == 4:
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
