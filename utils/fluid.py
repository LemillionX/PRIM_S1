import sys
sys.path.insert(0, '../src')
import tensorflow as tf
import numpy as np
import tf_solver_staggered as slv

class Fluid():
    def __init__(self):
        
        self.size = 20
        self.u = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.v = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.d = tf.zeros([self.size*self.size], dtype=tf.float32)

        # Training settings
        self.learning_rate = 1.5
        self.weight = 1

        # Simulation settings
        self.timestep = 0.025
        self.grid_min = -1
        self.grid_max = 1
        self.h = (self.grid_max - self.grid_min)/self.size
        self.diffusion_coeff = 0.0
        self.dissipation_rate = 0.0
        self.viscosity = 0.0
        self.boundary = "neumann"
        self.source = None

        # Intermediate variables
        self.coordsX = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.coordsY = tf.zeros([self.size*self.size], dtype=tf.float32)

    def setCoords(self):
        coordsX = np.zeros(self.size*self.size)
        coordsY = np.zeros(self.size*self.size)
        for j in range(self.size):
            for i in range(self.size):
                coordsX[i+j*self.size] = self.grid_min + (i+0.5)*self.h
                coordsY[i+j*self.size] = self.grid_min + (j+0.5)*self.h
        self.coordsX = tf.convert_to_tensor(coordsX, dtype=tf.float32)
        self.coordsY = tf.convert_to_tensor(coordsY, dtype=tf.float32)

    def setSize(self, size):
        self.size = size
        self.u = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.v = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.d = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.h = (self.grid_max - self.grid_min)/self.size
        self.setCoords()
