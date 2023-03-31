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

SIZE_X = 7          # number of elements in the x-axis
SIZE_Y = 7          # number of elements in the y-axis
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
        # v_init.append(np.sin(2*np.pi*point_x))
        
u_init = [-4.7420622e-07,  1.7267907e-04,  5.2999547e-03,  1.3012774e-05,
  5.5946828e-08,  1.7284689e-05, -2.3730587e-14,  6.3503256e-05,
  6.8785049e-02,  4.8146260e-01, -1.7692605e-01, -2.5006002e-02,
  3.9580908e-02,  1.9567315e-06,  7.7391338e-09,  8.5350173e-04,
 -1.0976689e+00, -1.2259898e+00, -1.2943808e+00,  5.7852726e-02,
 -1.1408865e-06, -5.5972706e-08, -5.1033620e-02, -1.5037079e-01,
  7.9767955e-03,  6.1002117e-02,  6.5998118e-03, -2.1722023e-08,
 -1.5790938e-07, -3.9015529e-01,  1.0172714e+00,  1.3087080e+00,
  1.1632949e+00, -2.8183435e-03, -2.7286458e-07, -2.0354046e-06,
 -4.5403585e-01, -2.2296385e-01,  3.1765860e-01, -3.1067733e-02,
  1.2492767e-02,  1.3650305e-06,  0.0000000e+00,  0.0000000e+00,
  0.0000000e+00,  1.6428115e-07,  2.3070558e-03,  3.2855656e-05,
  0.0000000e+00]

v_init = [1.05698757e-06,  2.72860314e-04,  3.72425281e-03, -1.33248119e-04,
  2.53480792e-08,  7.98247765e-06, -1.48284934e-15,  2.81712099e-04,
  6.49059862e-02,  3.14671695e-01, -4.49668288e-01, -8.06698918e-01,
  5.03685772e-02, -9.13346355e-07,  1.87931192e-07,  1.75960164e-03,
  1.71696758e+00,  3.60051766e-02, -1.50738072e+00, -4.27732319e-02,
 -1.23483942e-05,  1.12489026e-07,  4.00370583e-02,  1.66753411e+00,
 -2.07547754e-01, -1.47289240e+00, -4.19419911e-03, -2.60929767e-07,
  1.81436405e-06,  2.30760306e-01,  1.88600481e+00, -1.60940792e-02,
 -1.62100172e+00,  2.08883197e-03, -1.45107606e-07,  2.17296838e-06,
  4.48548906e-02,  1.09761786e+00,  4.35689449e-01, -1.17810130e-01,
  5.44340583e-04, -1.41645205e-06,  0.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  1.25467159e-06, -4.22280340e-04, -9.46654836e-06,
  0.00000000e+00]  

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
    velocity_field_x, velocity_field_y, density_field = simulate(t, velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF)
    # Viz update
    u_viz = viz.tensorToGrid(velocity_field_x.numpy(), SIZE_X, SIZE_Y)
    v_viz = viz.tensorToGrid(velocity_field_y.numpy(), SIZE_X, SIZE_Y)
    Q.set_UVC(u_viz,v_viz)
    viz.draw_density(viz.tensorToGrid(density_field.numpy(), SIZE_X, SIZE_Y), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)