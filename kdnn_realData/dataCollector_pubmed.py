"""
Created on 6/5/18

@author: Jingchao Yang
"""
import pandas as pd

allNews = pd.read_csv('yelpData/pubmed_Separate.csv')

categories = allNews['CATEGORY']
lat, lon = allNews['LAT'], allNews['LON']

cates, coors = [],[]
for i in range(len(categories)):
    cate = []
    for j in categories[i].split(','):
        cate.append(j)
    cates.append(cate)
    coors.append((lat[i],lon[i]))

# print(cates)