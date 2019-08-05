# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 20:38:11 2019

@author: Zoya
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class ClassK():
    
    def __init__(self, p, class_name = -1):
        self.p = p
        self.class_name = class_name
        
    def __getitem__(self, key):
        return self.p[key]
    
    def __len__(self):
        return len(self.p)
    
    @property
    def patterns(self):
        return self.p
    
    @property
    def closest(self):
        closes = 1    
        if len(self.p) == 0:
            return 100
        closes = []
        for p_template in self.p[:-1]:
            for p in self.p[:-1]:
                closes.append(distance.euclidean(p_template ,p))
        return np.median(np.asarray(closes))/ (len(self.p)**2)
    
    def plot(self, count = None, color = 'b', save = False):
        if count!=None and  len(self.p) >= count :
            count = count
        else:
            count = len(self.p)
        if count <= 1:
            return
        
        fig, axs = plt.subplots(count, 1, constrained_layout=False, figsize = (6,1*count))
        for i in range(count): 
            axs[i].plot(list(self.p[i]), color = color);
        fig.suptitle('Class = %d, Pattern count = %d\n'%(self.class_name, count), fontsize=14)
        fig.tight_layout()
        if save == True:
            fig.savefig('Class_%d.png'%self.dclass)   
        plt.show();
        

class Clusters(object):
    
    def __init__(self, data, labels, recunstruct):
        self.data = data
        self.labels = np.asarray([y for y in labels for i in range(recunstruct)])
        self.recunstruct = recunstruct
    @property
    def labels_var(self):
        return list(set(self.labels))
    
    def __getitem__(self, key):
        
        mask = self.labels == key
        indx = np.where(mask)[0]  
        count = len(indx)//self.recunstruct
        patterns = []
        for i in range(count-1):             
            part = indx[i*self.recunstruct:(i+1)*self.recunstruct]
            pattern = list(self.data[part])
            patterns.append(pattern)
      #  print("PPP: ", patterns)
        return ClassK(patterns, key)
    
    @property
    def patterns(self):
        classes = list(set(self.labels))  
        patterns = []
        for dclass in classes:
            pattern = self.__getitem__(dclass)
            if len(pattern)>1:
                patterns.append(pattern)
        return patterns            
    
    @property
    def sorted_patterns(self):  
        return sorted(self.patterns, key  = lambda p: p.closest)
    
    def pattern_plot(self):        
        labels  = list(set(self.labels))
        count_of_labels = len(labels)
        plt.figure(figsize = (24,4))
        for key in labels:
            mask = self.labels == key
            indx = np.where(mask)[0]  
            count = len(indx)//self.recunstruct
            
            t = key/count_of_labels
            c = [t,
                 t,
                 t]

            for i in range(count-1):             
                part = indx[i*self.recunstruct:(i+1)*self.recunstruct]
                plt.plot(part, self.data[part], color = c)
        return
            
    