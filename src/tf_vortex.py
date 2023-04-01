import tensorflow as tf
import tf_solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


##############################################################
#               General settings
##############################################################

SIZE_X = 25          # number of elements in the x-axis
SIZE_Y = 25          # number of elements in the y-axis
TIMESTEP = 0.025
N_FRAMES = 100     # number of frames to draw
GRID_MIN = -1
GRID_MAX = 1
D = (GRID_MAX-GRID_MIN)/SIZE_X

## Scalar field settings
K_DIFF = 0.0   #diffusion constant
ALPHA = 0.0       #dissipation rate

## Velocitity fiels settings
VISC = 0.0
VORTEX_RADIUS = 0.5*(GRID_MAX - GRID_MIN)/2.0
VORTEX_CENTER = np.array([GRID_MIN + D*(int(SIZE_X/2) + 0.5), GRID_MIN + D*(int(SIZE_Y/2)+0.5)  ])
#################################################################
# Initialisation
#################################################################
# Setup grids
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
u_init = []     # x-coordinates of speed
v_init = []     # y-coordinates of speed
density_init = []    # density field
for j in range(SIZE_Y):
    for i in range(SIZE_X):
        point_x = GRID_MIN+(i+0.5)*D
        point_y = GRID_MIN+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)
        r = np.linalg.norm( np.array([point_x, point_y]) - VORTEX_CENTER )
        if r < VORTEX_RADIUS:
            u_init.append(5*point_y)
            v_init.append(-5*point_x)
            density_init.append(1.0)
        else:
            u_init.append(0)
            v_init.append(0)
            density_init.append(0.0)

laplace_mat =  tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D)), dtype=tf.float32)
velocity_diff_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D) ), dtype=tf.float32)
scalar_diffuse_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D) ), dtype=tf.float32)

##############################################################
#               Plot initialisation 
##############################################################
x,y = np.meshgrid(COORDS_X[:SIZE_X], COORDS_Y[::SIZE_X])
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
Q = ax.quiver(x, y, viz.tensorToGrid(u_init, SIZE_X, SIZE_Y), viz.tensorToGrid(v_init, SIZE_X, SIZE_Y), color='red', scale_units='width')


## Converting variables to tensors
velocity_field_x = tf.convert_to_tensor(u_init, dtype=tf.float32)
velocity_field_y = tf.convert_to_tensor(v_init, dtype=tf.float32)
density_field = tf.convert_to_tensor(density_init, dtype=tf.float32)
dt = tf.convert_to_tensor(TIMESTEP, dtype=tf.float32)
COORDS_X = tf.convert_to_tensor(COORDS_X, dtype=tf.float32)
COORDS_Y = tf.convert_to_tensor(COORDS_Y, dtype=tf.float32)

##############################################################
#               Plot Animation
#############################################################
OUTPUT_DIR = "output"
FOLDER_NAME = "test_training_vel"
DENSITY_NAME = "test_training_dens"
DIR_PATH = os.path.join(os.getcwd().rsplit("\\",1)[0], OUTPUT_DIR)
SAVE_PATH =  os.path.join(DIR_PATH, FOLDER_NAME)
FPS = 20
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if not os.path.isdir(os.path.join(DIR_PATH, DENSITY_NAME)):
    os.mkdir(os.path.join(DIR_PATH, DENSITY_NAME))

print(SAVE_PATH)
pbar = tqdm(range(1, N_FRAMES+1), desc = "Simulating....")
plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(0)))
for t in pbar:
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF,boundary_func=None)
    # Viz update
    u_viz = tf.reshape(velocity_field_x, shape=(SIZE_X, SIZE_Y)).numpy()
    v_viz = tf.reshape(velocity_field_y, shape=(SIZE_X, SIZE_Y)).numpy()
    Q.set_UVC(u_viz,v_viz)
    viz.draw_density(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy(), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)