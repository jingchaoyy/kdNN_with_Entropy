""" Generating random points within a circular area
Author: Jingchao Yang
Date: Apr.1 2018
"""

import random
import math

# radius of the circle
circle_r = 100
# center of the circle (x, y)
circle_x = 5
circle_y = 7

def genPoints(numPs):
    points = []
    for i in range (numPs):
        point = []
        # random angle
        alpha = 2 * math.pi * random.random()
        # random radius
        r = circle_r * random.random()
        # calculating coordinates
        x = r * math.cos(alpha) + circle_x
        y = r * math.sin(alpha) + circle_y
        point.append(x)
        point.append(y)
        points.append(point)


    return points


# print("Random points", genPoints(100))


""" generate random world point geometries

def newpoint():
   return uniform(-180,180), uniform(-90, 90)

points = (newpoint() for x in xrange(10))
for point in points:
   print point
"""