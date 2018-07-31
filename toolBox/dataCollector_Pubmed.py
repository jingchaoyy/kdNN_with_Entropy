"""
Created on 6/5/18

@author: Jingchao Yang
"""
import pandas as pd

allNews = pd.read_csv('Data/pubmed_Separate.csv')

categories = allNews['CATEGORY']
lat, lon = allNews['LAT'], allNews['LON']

allCategories, allPoints = [],[]
for i in range(len(categories)):
    cate = []
    for j in categories[i].split(','):
        cate.append(j)
    allCategories.append(cate)
    allPoints.append((lat[i],lon[i]))

# print(allCategories)
