"""
Created on 6/6/18

@author: Xu Teng
"""
import random
import math


def randomData(num=100, radius=500, query_x=0, query_y=0, catNum=5, catRange=26):
    # num: Num of Entry
    # radius: Range of query
    # query_lat, query_lng: Location of query point
    # catNum: max Number of categories for each entry
    # catRange: types of categories
    allCategories = []
    allPoints = []
    # catSum = list(string.ascii_uppercase)
    for i in range(num):
        catTmp = set()
        alpha = 2 * math.pi * random.random()
        r = radius * random.random()
        loc_x = r * math.cos(alpha) + query_x
        loc_y = r * math.sin(alpha) + query_y
        allPoints.append((loc_x, loc_y))
        while len(catTmp) < catNum:
            catTmp.add(random.randint(0, catRange))
        allCategories.append(list(catTmp))
    # print(allPoints)
    print(allCategories)
    return allPoints, allCategories


def main():
    randomData()


if __name__ == "__main__":
    main()
