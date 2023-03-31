import tensorflow as tf
import numpy as np

def tensorToGrid(u, sizeX, sizeY):
    grid = np.zeros((sizeX, sizeY))
    for j in range(sizeY):
        for i in range(sizeX):
            grid[j,i] = u[i + sizeX*j]
    return grid

size = 5

# Create a tensor of shape (m)
tensor = tf.range(size*size)

# Define the value of n
n = size

# Compute a mask where every element whose index is 0 % n is True
mask = tf.logical_or(tf.logical_or(tf.math.equal(tf.range(tf.shape(tensor)[0]) % n, 0), tf.math.equal(tf.range(tf.shape(tensor)[0]) % n, n-1)),
                     tf.logical_or(tf.math.equal(tf.range(tf.shape(tensor)[0]) // n, 0), tf.math.equal(tf.range(tf.shape(tensor)[0]) // n, n-1)))

print(tensorToGrid(mask, size, size))  # [1 4 7 10]



