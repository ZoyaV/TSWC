# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:35:16 2019

@author: Zoya
"""

from scipy.spatial import distance
import numpy as np

def dist(p, threshold = 4):
#    closes = 1    
#    if len(p) <= threshold:
#        return 1000
    closes = []
    for p_template1 in p:
        for p_template2 in p:
#            r =  fastdtw(p_template1 ,p_template2, dist=distance.euclidean)[0]
            r = distance.euclidean(p_template1 ,p_template2)
            closes.append(r)
    return closes

def centr_dist(p):
     median_value = np.median(p, axis = 0)
     closes = []
     for p_template1 in p:
         r = distance.euclidean(p_template1 ,median_value)
         closes.append(r)
     return closes
 
