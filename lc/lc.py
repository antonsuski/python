
import sys
import math
import random
import numpy
import matplotlib.pyplot as plt

class vec2:
    def __init__(self, x,y):
        self.x = x
        self.y = y

def basic_linear_classifier(set1, set2):
    return numpy.dot(vec, hs)

def make_rand_set(min = -3, max = 3, precision = 2, size = 5):
    tmp_buff = []
    for i in range(size):
        x = round(random.uniform(min, max), precision)
        y = round(random.uniform(min, max), precision)
        tmp_buff.append(vec2(x,y));
    return tmp_buff

D0 = make_rand_set(0, 2)
D1 = make_rand_set(0, 2)
D0.insert(0, (1,1))
D1.insert(0, (1,1))

print(D0, D1)
# print(D0, D1, basic_linear_classifier(D0, D1))
