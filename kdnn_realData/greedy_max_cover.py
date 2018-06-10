"""
Created on 6/10/18

@author: Xu Teng
"""
import csv
import math


def main():
    ###query arguments###
    query_x = 0
    query_y = 0
    query_range = 40
    dataset = './Data/news.csv'
    k = 10
    #####################

    catSum = set()
    subsetList = []
    indexList = []
    count = 0

    with open(dataset) as mycsv:
        reader = csv.reader(mycsv, delimiter='\t')
        for row in reader:
            distance = math.sqrt((float(row[2]) - query_x) * (float(row[2]) - query_x) + (float(row[3]) - query_y) * (
            float(row[3]) - query_y))
            if (distance <= query_range):
                catList = row[1].split("|")
                subsetList.append(set(catList))
                indexList.append(count)
                for element in catList:
                    catSum.add(element)
            count = count + 1

        ####Greedy ALG#####
    res = []
    if (k >= len(subsetList)):
        print(indexList)
    else:
        covered = set()
        cover = []
        for i in range(k):
            maxSub = max(subsetList, key=lambda s: len(s - covered))
            res.append(indexList[subsetList.index(maxSub)])
            covered |= maxSub
        print(res)


if __name__ == '__main__':
    main()
