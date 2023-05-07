import np_solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


##############################################################
#               General settings
##############################################################

SIZE_X = 70          # number of elements in the x-axis
SIZE_Y = 70          # number of elements in the y-axis
TIMESTEP = 0.1
N_FRAMES = 600     # number of frames to draw
GRID_MIN = 0
GRID_MAX = 2
D = (GRID_MAX-GRID_MIN)/SIZE_X

## Scalar field settings
K_DIFF = 0.0   #diffusion constant
ALPHA = 0.0       #dissipation rate

## Velocitity fiels settings
VISC = 0.0
VORTEX_RADIUS = 3/15
VORTEX_CENTER = np.array([GRID_MAX/4, GRID_MAX/2])
KARMAN_VELOCITY = 0.5
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
        v_init.append(0)
        u_init.append(KARMAN_VELOCITY)
        density_init.append(0.0)

## Initialise variables
laplace_mat = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D))
velocity_diff_mat = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D) )
scalar_diffuse_mat = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D) )
velocity_field_x = np.array(u_init)
velocity_field_y = np.array(v_init)
density_field =density_init
dt = TIMESTEP

def karman_boundary(_u, _v, _sizeX, _sizeY,_b=0):
    new_u = np.copy(_u)
    new_v = np.copy(_v)
    for i in range(_sizeX):
        for j in range(_sizeY):
            if (i ==0):
                new_u[slv.indexTo1D(i,j,_sizeX)] = KARMAN_VELOCITY
                new_v[slv.indexTo1D(i,j,_sizeX)] = 0
            if (j==_sizeY-1):
                new_u[slv.indexTo1D(i,j,_sizeX)] = 0
                new_v[slv.indexTo1D(i,j,_sizeX)] = 0
            if (j==0):
                new_u[slv.indexTo1D(i,j,_sizeX)] = 0
                new_v[slv.indexTo1D(i,j,_sizeX)] = 0
            if (i==_sizeX-1):
                # new_u[slv.indexTo1D(i,j,_sizeX)] = _u[slv.indexTo1D(i-1,j,_sizeX)]
                # new_v[slv.indexTo1D(i,j,_sizeX)] = _v[slv.indexTo1D(i-1,j,_sizeX)]
                new_v[slv.indexTo1D(i,j,_sizeX)] = 0
            point_x = GRID_MIN+(i+0.5)*D
            point_y = GRID_MIN+(j+0.5)*D
            r = np.linalg.norm( np.array([point_x, point_y]) - VORTEX_CENTER )
            if r < VORTEX_RADIUS:
                new_u[slv.indexTo1D(i,j,_sizeX)] = 0
                new_v[slv.indexTo1D(i,j,_sizeX)] = 0
    return new_u, new_v
boundary_func = karman_boundary
##############################################################
#               Plot initialisation 
##############################################################
# x,y = np.meshgrid(COORDS_X[:SIZE_X], COORDS_Y[::SIZE_X])
# fig, ax = plt.subplots(1, 1)
# ax.set_aspect('equal', adjustable='box')
# Q = ax.quiver(x, y, viz.tensorToGrid(u_init, SIZE_X, SIZE_Y), viz.tensorToGrid(v_init, SIZE_X, SIZE_Y), color='red', scale_units='width')



##############################################################
#               Plot Animation
#############################################################
OUTPUT_DIR = "output"
FOLDER_NAME = "karman_velocity"
DENSITY_NAME = "karman_density"
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
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF, boundary_func)
    # Viz update
    # viz.draw_density(viz.tensorToGrid(density_field, SIZE_X, SIZE_Y), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    u_viz = viz.tensorToGrid(velocity_field_x, SIZE_X, SIZE_Y)
    v_viz = viz.tensorToGrid(velocity_field_y, SIZE_X, SIZE_Y)
    curl = viz.curl(u_viz.T, v_viz.T)
    viz.draw_curl(curl.T, os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    # Q.set_UVC(u_viz,v_viz)
    # plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

# viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)