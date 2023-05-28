import tensorflow as tf
import tf_solver_staggered as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


##############################################################
#               General settings
##############################################################

SIZE_X = 55          # number of elements in the x-axis
SIZE_Y = 55          # number of elements in the y-axis
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
VORTEX_CENTER = np.array([0,0])
VORTEX_MAGNITUDE = 10
#################################################################
# Initialisation
#################################################################
RESOLUTION_LIMIT = 30
# Setup grids
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
u_init = np.zeros(SIZE_X*SIZE_Y)     # x-coordinates of speed
v_init = np.zeros(SIZE_X*SIZE_Y)      # y-coordinates of speed
density_init = np.zeros(SIZE_X*SIZE_Y)    # density field
for j in range(SIZE_Y):
    for i in range(SIZE_X):
        point_x = GRID_MIN+(i+0.5)*D
        point_y = GRID_MIN+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)
        r = np.linalg.norm( np.array([point_x, point_y]) - VORTEX_CENTER )
        r_u = np.linalg.norm( np.array([point_x-0.5*D, point_y]) - VORTEX_CENTER )
        r_v = np.linalg.norm( np.array([point_x, point_y-0.5*D]) - VORTEX_CENTER )
        if r_u < VORTEX_RADIUS:
            u_init[i+j*SIZE_X] = VORTEX_MAGNITUDE*point_y
        if r_v < VORTEX_RADIUS:
            v_init[i+j*SIZE_X] = -VORTEX_MAGNITUDE*point_x
        if r < VORTEX_RADIUS:
            density_init[i+j*SIZE_X] = 1.0

## Initialise variables
laplace_mat =  tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D)), dtype=tf.float32)
velocity_diff_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D) ), dtype=tf.float32)
scalar_diffuse_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D) ), dtype=tf.float32)
velocity_field_x = tf.convert_to_tensor(u_init, dtype=tf.float32)
velocity_field_y = tf.convert_to_tensor(v_init, dtype=tf.float32)
density_field = tf.convert_to_tensor(density_init, dtype=tf.float32)
dt = tf.convert_to_tensor(TIMESTEP, dtype=tf.float32)
COORDS_X = tf.convert_to_tensor(COORDS_X, dtype=tf.float32)
COORDS_Y = tf.convert_to_tensor(COORDS_Y, dtype=tf.float32)

##############################################################
#               Plot initialisation 
##############################################################
x,y = np.meshgrid(COORDS_X[:SIZE_X], COORDS_Y[::SIZE_X])
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
vel_x, vel_y = slv.velocityCentered(velocity_field_x,velocity_field_y, SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, GRID_MIN, D)
Q = ax.quiver(x, y, tf.reshape(vel_x, shape=(SIZE_X, SIZE_Y)).numpy(), tf.reshape(vel_y, shape=(SIZE_X, SIZE_Y)).numpy(), color='red', scale_units='width')

##############################################################
#               Plot Animation
#############################################################
OUTPUT_DIR = "output"
FOLDER_NAME = "vortex_velocity_"+str(SIZE_X)+"x"+str(SIZE_Y)
DENSITY_NAME = "vortex_density_"+str(SIZE_X)+"x"+str(SIZE_Y)
DIR_PATH = os.path.join(os.getcwd().rsplit("\\",1)[0], OUTPUT_DIR)
SAVE_PATH =  os.path.join(DIR_PATH, FOLDER_NAME)
FPS = 20
if SIZE_X < RESOLUTION_LIMIT:
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
viz.draw_density(np.flipud(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy()), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(0)))
if SIZE_X < RESOLUTION_LIMIT:
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(0)))
for t in pbar:
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF)
    # Viz update
    viz.draw_density(np.flipud(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy()), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    if SIZE_X < RESOLUTION_LIMIT:
        vel_x, vel_y = slv.velocityCentered(velocity_field_x,velocity_field_y, SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, GRID_MIN, D)
        u_viz = tf.reshape(vel_x, shape=(SIZE_X, SIZE_Y)).numpy()
        v_viz = tf.reshape(vel_y, shape=(SIZE_X, SIZE_Y)).numpy()
        Q.set_UVC(u_viz,v_viz)
        plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

if SIZE_X < RESOLUTION_LIMIT:
    viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)