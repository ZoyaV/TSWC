# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:39:41 2019

@author: Zoya
"""


import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.cluster import KMeans

import sys
sys.path.append('../ikmeans')

from distations import centr_dist
from clusters import Clusters

def centnters_update(centers, split_mask = None):
    new_centers = []
    if split_mask is not None:
        for c,split in zip(centers, split_mask):
            if split:
                new_centers+=[c,c]
            else:
                new_centers+=[c]
    else:   
        for c in centers:
            new_centers+=[c,c]    
    return np.asarray(new_centers)

class IKMeans():
    
    filter_type = 'haar'
    def __init__(self, data, k = 3, start_lvl = None, end_lvl = 1, union_level = False):       
        self.max_lvl = pywt.dwt_max_level(data_len= len(data),
                                             filter_len= self.filter_type)
        
        self.data = data
        self.start_lvl = start_lvl
        self.end_lvl = end_lvl
        self.curr_lvl = start_lvl
        self.data = data
        self.cur_cA = None
        self.cD = None
        self.centers = None
        self.labels = None
        self.recunstruct = 2**self.curr_lvl 
        self.k = k
        self.clasters = None
        self.union_level = union_level
        
        if start_lvl is not None:
            
            coeffs = pywt.wavedec(data, self.filter_type, level = self.start_lvl)
            self.cur_cA, self.cD =  coeffs[0], coeffs[1:] 
        else:            
            coeffs = pywt.wavedec(data, self.filter_type)
            self.cur_cA, self.cD =  coeffs[0], coeffs[1:]  
        
    def next_lvl(self)->None:
        i = self.start_lvl - self.curr_lvl 
        self.cur_cA = self.cur_cA + self.cD[i]  
        self.cur_cA  = pywt.upcoef('a',  self.cur_cA, self.filter_type, take = len(self.cD[i + 1]))
        self.curr_lvl-=1
        
        self.recunstruct = 2**self.curr_lvl
        self.k*=2
        return
    def __claster_dencity(self, i):
        t = self.start_lvl - self.curr_lvl 
        mask = np.where(self.labels == i)
        X = np.asarray(list(zip(self.cur_cA, self.cD[t])))        
        claster = X[mask]
        mean_d =np.var(claster)
        return mean_d
    
    def __clasters_dencity(self):
        max_claster_idx = max(list(set(self.labels))) + 1
        clater_idxs = list(set(self.labels))
        dincities = [0] * max_claster_idx
        for i in clater_idxs:
#            print(i)
#            print(dincities[i])
            dincities[i] = self.__claster_dencity(i)
        return np.asarray(dincities)
    
    @property
    def __split_mask(self):
        dincities = self.__clasters_dencity()
        split_mask = np.zeros_like(dincities)
        threshold = np.mean(dincities)
        
        for i,d in enumerate(dincities):
            split_mask[i] = d > threshold
        return split_mask     
            
            
    def fit(self):
        i = self.start_lvl - self.curr_lvl
        X = np.asarray(list(zip(self.cur_cA, self.cD[i])))
        if self.centers is not None:
            if self.union_level:
                self.centers = centnters_update(self.centers, self.__split_mask)
            else:
                self.centers = centnters_update(self.centers)
            self.k = len(self.centers)
            kmeans = KMeans(n_clusters= self.k, init = self.centers)
        else:
            kmeans = KMeans(n_clusters= self.k)     

        kmeans.fit(X)
        self.labels = kmeans.predict(X)
        self.centers = kmeans.cluster_centers_
        self.clasters = Clusters(self.data, self.labels, self.recunstruct)
        
        return Clusters(self.data, self.labels, self.recunstruct)
    
    def plot_clusters(self, show_number = True):
        plt.figure(figsize = (10,10))
        i = self.start_lvl - self.curr_lvl
        plt.scatter(self.cur_cA, self.cD[i], c=self.labels, s=10, cmap='viridis')
        centers = self.centers
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=20, alpha=0.5);
        plt.title("Decomposition level = %d, clasters = %d"%(self.curr_lvl, self.k))
        plt.xlabel('Aproximation')
        plt.ylabel('Details')
        if show_number:
            for i,center in enumerate(centers):
                plt.text(*center, str(i), fontsize=12)
        plt.show();