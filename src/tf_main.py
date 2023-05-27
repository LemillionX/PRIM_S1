import tensorflow as tf
import tf_solver as slv
import tf_train as train
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
# SOURCE = {}
# SOURCE["value"]=1.0
# indices = np.where(np.array(density_init) == 1.0)[0]
# SOURCE["indices"]= indices.reshape((np.shape(indices)[0], 1))
# SOURCE["time"]=1

#####################
# VORTICES SETTINGS #
#####################
# NB_VORTICES = 2
# COORDS = tf.stack((COORDS_X, COORDS_Y), axis=1)
# centers = tf.convert_to_tensor([[0.5, -1.2], [-0.5, -0.5], [0, 0.5], [-0.4, 0.4], [-1, 0.25],  [-0.4, 0.3] ]  , dtype=tf.float32)
# radius = 0.2*tf.convert_to_tensor([2.5, 2, 1.5, 1, 1.5, 1.5]  , dtype=tf.float32)
# w = 30*tf.convert_to_tensor([-2, 2.5, -5, 1, 2.5, -7]  , dtype=tf.float32)
# u_init,v_init = train.init_vortices(NB_VORTICES, centers, radius, w, COORDS, SIZE_X)

laplace_mat =  tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D)), dtype=tf.float32)
velocity_diff_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D) ), dtype=tf.float32)
scalar_diffuse_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D) ), dtype=tf.float32)

##############################################################
#               Plot initialisation 
##############################################################
x,y = np.meshgrid(COORDS_X[:SIZE_X], COORDS_Y[::SIZE_X])
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
Q = ax.quiver(x, y, tf.reshape(u_init, shape=(SIZE_X, SIZE_Y)).numpy(), tf.reshape(v_init, shape=(SIZE_X, SIZE_Y)).numpy(), color='red', scale_units='width')

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
else:
    for file in os.listdir(SAVE_PATH):
        file_path = os.path.join(SAVE_PATH, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Error deleting file: {file_path} - {e}')
if not os.path.isdir(os.path.join(DIR_PATH, DENSITY_NAME)):
    os.mkdir(os.path.join(DIR_PATH, DENSITY_NAME))
else:
    for file in os.listdir(os.path.join(DIR_PATH, DENSITY_NAME)):
        file_path = os.path.join(os.path.join(DIR_PATH, DENSITY_NAME), file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Error deleting file: {file_path} - {e}')



print(SAVE_PATH)
pbar = tqdm(range(1, N_FRAMES+1), desc = "Simulating....")
plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(0)))
viz.draw_density(np.flipud(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy()), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(0)))

for t in pbar:
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF,boundary_func=None, source=SOURCE, t=t)
    # Viz update
    u_viz = tf.reshape(velocity_field_x, shape=(SIZE_X, SIZE_Y)).numpy()
    v_viz = tf.reshape(velocity_field_y, shape=(SIZE_X, SIZE_Y)).numpy()
    Q.set_UVC(u_viz,v_viz)
    viz.draw_density(np.flipud(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy()), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)