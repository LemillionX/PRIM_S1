import tensorflow as tf
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt

def create_vortex(center, r, w, coords):
    rel_coords = coords - center
    dist = tf.linalg.norm(rel_coords, axis=-1)
    mask = tf.where(tf.greater(dist, r))
    u = w*rel_coords[..., 1]
    v = -w*rel_coords[..., 0]
    u = tf.expand_dims(u, 1)
    v = tf.expand_dims(v, 1)
    u = tf.tensor_scatter_nd_update(u, mask, tf.zeros_like(mask, dtype=tf.float32))
    v = tf.tensor_scatter_nd_update(v, mask, tf.zeros_like(mask, dtype=tf.float32))
    return u,v

# Calculate some useful physicial quantities
GRID_MAX = 1
GRID_MIN = -1
SIZE = 25
D = (GRID_MAX - GRID_MIN)/SIZE
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position

for j in range(SIZE):
    for i in range(SIZE):
        point_x = GRID_MIN+(i+0.5)*D
        point_y = GRID_MIN+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)

# Create vortices
COORDS = tf.stack((COORDS_X, COORDS_Y), axis=1)
center = tf.Variable(tf.random.normal([2]), dtype=tf.float32)
r = tf.Variable(0.5*tf.abs(tf.random.normal([1])), dtype=tf.float32)
w = tf.Variable(D*tf.random.normal([1]), dtype=tf.float32)
print("center = ", center)
print("r = ", r)
print("w = ", w)

# Gradient
with tf.GradientTape() as tape:
    u,v = create_vortex(center,r,w,COORDS)
grad_generating = tape.gradient([u,v], [center, r, w])

# Differentiability test
if all([gradient is not None for gradient in grad_generating]):
    print(colored("create_vortex is differentiable.", 'green'))
else:
    print(colored("create_vortex is not differentiable.", 'red'))
    print(grad_generating)

# Affichage
x,y = np.meshgrid(COORDS_X[:SIZE], COORDS_Y[::SIZE])
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
u_viz = tf.reshape(u, shape=(SIZE, SIZE)).numpy()
v_viz = tf.reshape(v, shape=(SIZE, SIZE)).numpy()
Q = ax.quiver(x, y, u, v, color='red', scale_units='width')
plt.show()
