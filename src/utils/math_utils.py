import math
import numpy as np

def fexp(x, p):
    return np.sign(x) * (np.abs(x) ** p)

def calculate_3d_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance
