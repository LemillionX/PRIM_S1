import tensorflow as tf
import tf_solver_staggered as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json

##############################################################
#               General settings
##############################################################

SIZE_X = 20          # number of elements in the x-axis
SIZE_Y = 20          # number of elements in the y-axis
TIMESTEP = 0.025
N_FRAMES = 200     # number of frames to draw
GRID_MIN = -1
GRID_MAX = 1
D = (GRID_MAX-GRID_MIN)/SIZE_X
BOUNDARY = "neumann"

## Scalar field settings
K_DIFF = 0.0   #diffusion constant
ALPHA = 0.0       #dissipation rate

## Velocitity fiels settings
VISC = 0.0

RESOLUTION_LIMIT = 30

####################################################################
#   Loading Files
####################################################################
# Load data from .json file
CONSTRAINT = {}
CONSTRAINT_FILE = "snake"
with open("../data/"+CONSTRAINT_FILE+".json") as file:
    print('Loading file', CONSTRAINT_FILE+".json")
    CONSTRAINT = json.load(file)

# Data always have to contain target and initial density
density_init = CONSTRAINT["init_density"][:]

#####################
# SOURCE SETTINGS   #
#####################
SOURCE = None
SOURCE = {}
SOURCE["value"]=1.0
indices = np.where(np.array(density_init) == 1.0)[0]
SOURCE["indices"]= indices.reshape((np.shape(indices)[0], 1))
SOURCE["time"]= N_FRAMES

# Setup grids
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
u_init = np.zeros(SIZE_X*SIZE_Y)     # x-coordinates of speed
v_init = np.zeros(SIZE_X*SIZE_Y)      # y-coordinates of speed
for j in range(SIZE_Y):
    for i in range(SIZE_X):
        point_x = GRID_MIN+(i+0.5)*D
        point_y = GRID_MIN+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)

## Initialise variables
lu, p =  slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D), BOUNDARY)
velocity_diff_lu, velocity_diff_p = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D))
scalar_diffuse_lu, scalar_diffuse_p = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D))
velocity_field_x = tf.convert_to_tensor(u_init, dtype=tf.float32)
velocity_field_y = tf.convert_to_tensor(v_init, dtype=tf.float32)
density_field = tf.convert_to_tensor(density_init, dtype=tf.float32)
dt = tf.convert_to_tensor(TIMESTEP, dtype=tf.float32)
COORDS_X = tf.convert_to_tensor(COORDS_X, dtype=tf.float32)
COORDS_Y = tf.convert_to_tensor(COORDS_Y, dtype=tf.float32)

##############################################################
#               Plot Animation 
##############################################################
FPS = 20
OUTPUT_DIR = "output"
FILE_NAME = "buoyancy"+str(SIZE_X)+"x"+str(SIZE_Y)
V_PATH, D_PATH, RESOLUTION_LIMIT = viz.init_dir(OUTPUT_DIR, FILE_NAME, SIZE_X) 

fig, ax, Q = viz.init_viz(velocity_field_x,velocity_field_y,density_field, COORDS_X, COORDS_Y, SIZE_X, SIZE_Y, GRID_MIN, D, V_PATH, D_PATH, RESOLUTION_LIMIT)

pbar = tqdm(range(1, N_FRAMES+1), desc = "Simulating....")
for t in pbar:
    f_u, f_v = slv.buoyancyForce(density_field, SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, GRID_MIN, D)
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field, SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, lu, p, ALPHA, velocity_diff_lu, velocity_diff_p, VISC, scalar_diffuse_lu, scalar_diffuse_p, K_DIFF, boundary_func=BOUNDARY, source=SOURCE, t=t, f_u=f_u, f_v=f_v)
    # Viz update
    viz.draw_density(np.flipud(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy()), os.path.join(D_PATH, '{:04d}.png'.format(t)))
    if SIZE_X < RESOLUTION_LIMIT:
        u_viz, v_viz = viz.draw_velocity(velocity_field_x,velocity_field_y, SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, GRID_MIN, D)
        Q.set_UVC(u_viz,v_viz)
        plt.savefig(os.path.join(V_PATH, '{:04d}'.format(t)))

if SIZE_X < RESOLUTION_LIMIT:
    viz.frames2gif(V_PATH, V_PATH+".gif", FPS)
viz.frames2gif(D_PATH, D_PATH+".gif", FPS)