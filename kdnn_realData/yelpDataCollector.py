"""
Created on 5/14/18

@author: Jingchao Yang
"""
import json
import os


class Restaurant:
    name = "Restaurant"
    sta = 'state'
    categor = 'category'
    xy = ()

    def addName(self, new_name):
        self.name = new_name

    def addState(self, sta_Name):
        self.sta = sta_Name

    def addCategory(self, categories):
        self.categor = categories

    def addLocation(self, x, y):
        self.xy = (x, y)


# return all Restaurant in AZ
def dataPre(filePath):
    with open(os.getcwd() + filePath) as f:
        content = f.readlines()
        # print(content)

    allRest = []
    for record in content:
        rest = Restaurant()
        data = json.loads(record)
        rest.addName(data['name'])
        rest.addState(data['state'])
        rest.addCategory(data['categories'])
        rest.addLocation(data['latitude'], data['longitude'])

        allRest.append(rest)

    # Collecting restaurants in AZ
    AZ = []
    for RS in allRest:
        if RS.sta == 'AZ':
            AZ.append(RS)

    return AZ


# restaurants in AZ with coordinates from original datasets
STFY = dataPre("/yelpData/business.json")
allPoints = []
for i in STFY:
    allPoints.append(i.xy)

# print(allPoints)
