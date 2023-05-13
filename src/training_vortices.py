import tensorflow as tf
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import sys 

@tf.function
def create_vortex(center, r, w, coords, alpha=1.0):
    rel_coords = coords - center
    dist = tf.linalg.norm(rel_coords, axis=-1)
    smoothed_dist = tf.exp(-tf.pow((dist-r)*alpha/r,2.0))
    u = w*rel_coords[...,1] * smoothed_dist
    v = - w*rel_coords[..., 0] * smoothed_dist
    return u,v

@tf.function
def init_vortices(n, centers, radius, w, coords, size):
    u = tf.zeros([size*size])
    v = tf.zeros([size*size])
    for i in range(n):
        u_tmp,v_tmp = create_vortex(centers[i], radius[i], w[i], coords) 
        u += u_tmp
        v += v_tmp
    return u,v

