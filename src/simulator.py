import np_solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


##############################################################
#               General settings
##############################################################

SIZE_X = 19          # number of elements in the x-axis
SIZE_Y = 19          # number of elements in the y-axis
TIMESTEP = 0.025
N_FRAMES = 300     # number of frames to draw
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
            u_init.append(10*point_y)
            v_init.append(-10*point_x)
            density_init.append(1.0)
        else:
            u_init.append(0)
            v_init.append(0)
            density_init.append(0.0)
        # v_init.append(np.sin(2*np.pi*point_x))
        
laplace_mat = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D))
velocity_diff_mat = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D) )
scalar_diffuse_mat = slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D) )

##############################################################
#               Plot initialisation 
##############################################################
x,y = np.meshgrid(COORDS_X[:SIZE_X], COORDS_Y[::SIZE_X])
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
Q = ax.quiver(x, y, viz.tensorToGrid(u_init, SIZE_X, SIZE_Y), viz.tensorToGrid(v_init, SIZE_X, SIZE_Y), color='red', scale_units='width')


## Converting variables to tensors
velocity_field_x = np.array(u_init)
velocity_field_y = np.array(v_init)
density_field =density_init
dt = TIMESTEP

##############################################################
#               Solving
##############################################################
def simulate(_t, _u, _v, _s, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h, _mat, _alpha, _vDiff_mat, _visc, _sDiff_mat, _kDiff):
    ## Vstep
    # advection step
    new_u = slv.advectCentered(_u, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    new_v = slv.advectCentered(_v, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    _u = new_u
    _v = new_v

    # diffusion step
    if _visc > 0:
        _u = slv.diffuse(_u, _vDiff_mat)[..., 0]
        _v = slv.diffuse(_v, _vDiff_mat)[..., 0]

    # projection step
    _u, _v = slv.project(_u, _v, _sizeX, _sizeY, _mat, _h)

    ## Sstep
    # advection step
    _s = slv.advectCentered(_s, _u, _v, _sizeX, _sizeY, _coord_x, _coord_y, _dt, _offset, _h)
    
    # diffusion step
    if _kDiff > 0:
        _s = slv.diffuse(_s, _sDiff_mat)

    # dissipation step
    _s = slv.dissipate(_s, _alpha, _dt)
    return _u, _v, _s


##############################################################
#               Plot Animation
#############################################################
OUTPUT_DIR = "output"
FOLDER_NAME = "simul_velocity"
DENSITY_NAME = "simul_density"
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
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF)
    # Viz update
    viz.draw_density(viz.tensorToGrid(density_field, SIZE_X, SIZE_Y), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    u_viz = viz.tensorToGrid(velocity_field_x, SIZE_X, SIZE_Y)
    v_viz = viz.tensorToGrid(velocity_field_y, SIZE_X, SIZE_Y)
    Q.set_UVC(u_viz,v_viz)
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)