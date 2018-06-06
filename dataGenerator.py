"""
Created on 6/6/18

@author: Xu Teng
"""
import random
import math
import string


def randomData(num=100, radius=500, query_x=0, query_y=0, catNum=5, catRange=26):
    # num: Num of Entry
    # radius: Range of query
    # query_lat, query_lng: Location of query point
    # catNum: Number of categories for each entry
    # catRange: Range of categories
    catList = []
    locList = []
    catSum = list(string.ascii_uppercase)
    for i in range(num):
        catTmp = set()
        alpha = 2 * math.pi * random.random()
        r = radius * random.random()
        loc_x = r * math.cos(alpha) + query_x
        loc_y = r * math.sin(alpha) + query_y
        locList.append((loc_x, loc_y))
        while len(catTmp) < catNum:
            catTmp.add(catSum[random.randint(0, catRange - 1)])
        catList.append(list(catTmp))
    print(locList)
    print(catList)
    return locList, catList


def main():
    randomData()


if __name__ == "__main__":
    main()
