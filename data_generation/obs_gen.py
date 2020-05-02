# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:58:28 2020

@author: Kareem
"""
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as dist_transform
import matplotlib.pyplot as plt
import pylbm

def get_random_triangle(p1 = [0.3,0.3], p2 = [0.7,0.3], p3 = [0.5,0.7], var = 0.3):
    
    start = (np.random.rand(2)-0.5) * var + p1
    end1 = (np.random.rand(2)-0.5) * var + p2
    v1 = end1 - start
    end2 = (np.random.rand(2)-0.5) * var + p3
    v2 = end2 - start
    obs = pylbm.Triangle(start,v1,v2,label=2)
    return obs

def plot_random_triangle():
    plt.figure()

    x_search = np.linspace(0,2,128)
    y_search = np.linspace(0,1,64)
    
    obs = get_random_triangle()
    g,h = get_distance(x_search,y_search,obs)
    plt.imshow(h.T)
    

def get_distance(search_x, search_y, obs):
    dists = np.zeros((len(search_x),len(search_y)))
    presence_map = np.ones((len(search_x),len(search_y)))
    for i,x in enumerate(search_x):
        for j,y in enumerate(search_y):
            if j == 0 or j == len(search_y)-1:
                presence_map[i,j] = 1
            else:
                presence_map[i,j] = obs.point_inside((x,y))
            
           
    presence_map = presence_map.astype(bool)
    dists = dist_transform(np.invert(presence_map))- dist_transform(presence_map)
    return presence_map,dists