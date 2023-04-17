import tensorflow as tf
import numpy as np

FILENAME = {}
FILENAME["velocity"] = "trained_velocity_obstacle2"
FILENAME["density"] = "trained_density_obstacle2"

density_init=[
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0   
]

target_density =[
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,  
 0,0,0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0,0,0,  
 0,0,0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0,0,0,  
 0,0,0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0,0,0,  
 0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

MAX_ITER = 130
N_FRAMES = 80     # number of the frame where we want the shape to be matched
SIZE = int(np.sqrt(len(target_density)))
FLUID_SETTINGS = {}
FLUID_SETTINGS["timestep"] = 0.025
FLUID_SETTINGS["grid_min"] = -1
FLUID_SETTINGS["grid_max"] = 1
FLUID_SETTINGS["diffusion_coeff"] = 0.0
FLUID_SETTINGS["dissipation_rate"] = 0.0
FLUID_SETTINGS["viscosity"] = 0.0
FLUID_SETTINGS["source"] = {}
FLUID_SETTINGS["source"]["value"]=1.0
FLUID_SETTINGS["source"]["indices"]=[]
FLUID_SETTINGS["source"]["time"]=20

D = (FLUID_SETTINGS["grid_max"] -FLUID_SETTINGS["grid_min"])/SIZE
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
u_init = []     # x-coordinates of speed
v_init = []     # y-coordinates of speed

OBSTACLES = []
SOURCE_WIDTH = 4
OFFSET = (SIZE - SOURCE_WIDTH)//2
for j in range(SIZE):
    for i in range(SIZE):
        if target_density[i + j*SIZE] == 2:
            OBSTACLES.append(i+j*SIZE)
            target_density[i+j*SIZE] = 0
        if (OFFSET <= i < OFFSET + SOURCE_WIDTH) and (SIZE - (SOURCE_WIDTH+1) <= j <= SIZE - 2):
            FLUID_SETTINGS["source"]["indices"].append([i+j*SIZE])
        point_x = FLUID_SETTINGS["grid_min"]+(i+0.5)*D
        point_y = FLUID_SETTINGS["grid_min"]+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)
        u_init.append(0)
        v_init.append(-20)

def build_laplacian_matrix(sizeX,sizeY,a,b):
    mat = np.zeros((sizeX*sizeY,sizeX*sizeY))
    for it in range(sizeX*sizeY):
        i = it%sizeX
        j = it//sizeX
        mat[it,it] = b
        if (i>0):
            mat[it,it-1] = a
        if (i<sizeX-1):
            mat[it, it+1] = a
        if (j>0):
            mat[it,it-sizeX] = a
        if (j<sizeY-1):
            mat[it, it+sizeX] = a
        if (i+1 == sizeX - 1) or (i-1 == 0) or (j+1 == sizeY-1) or (j-1 == 0):
            mat[it, it] = b*3/4
    return mat

OBSTACLES = tf.expand_dims(tf.convert_to_tensor(OBSTACLES), 1)
FLUID_SETTINGS["source"]["indices"] = tf.convert_to_tensor(FLUID_SETTINGS["source"]["indices"])

import tf_solver as slv
import tf_train as train

@tf.function
def set_obstacles_boundary(u,v,sizeX,sizeY,b=0):
    new_u = tf.identity(u)
    new_v = tf.identity(v)
    mask_down = tf.expand_dims(tf.range(0, sizeX, 1),1)
    mask_up = tf.expand_dims(tf.range(sizeX*(sizeY-1), sizeX*sizeY, 1), 1)
    mask_left = tf.expand_dims(tf.range(0, sizeX*sizeY, sizeX),1)
    mask_right = tf.expand_dims(tf.range(sizeX-1,sizeX*sizeY,sizeX), 1)

    new_u = tf.tensor_scatter_nd_update(new_u, OBSTACLES, tf.constant(0, dtype=tf.float32, shape=[OBSTACLES.shape[0]]))
    new_v = tf.tensor_scatter_nd_update(new_v, OBSTACLES, tf.constant(0, dtype=tf.float32, shape=[OBSTACLES.shape[0]]))

    # Left boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_left, tf.abs(tf.gather_nd(tf.roll(u, shift=-1, axis=0), mask_left)))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_left, tf.gather_nd(tf.roll(v, shift=-1, axis=0), mask_left))
    # Right boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_right, -tf.abs(tf.gather_nd(tf.roll(u, shift=1, axis=0), mask_right)))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_right, tf.gather_nd(tf.roll(v, shift=1, axis=0), mask_right))
    # Up boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_up, tf.gather_nd(tf.roll(u, shift=sizeX, axis=0), mask_up))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_up, -tf.abs(tf.gather_nd(tf.roll(v, shift=sizeX, axis=0), mask_up)))
    # Down boundary
    new_u = tf.tensor_scatter_nd_update(new_u, mask_down, tf.gather_nd(tf.roll(u, shift=-sizeX, axis=0), mask_down))
    new_v = tf.tensor_scatter_nd_update(new_v, mask_down, tf.abs(tf.gather_nd(tf.roll(v, shift=-sizeX, axis=0), mask_down)))
    
    # Upper-left corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[slv.indexTo1D(0, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_u[slv.indexTo1D(0, sizeY-2, sizeX)] + new_u[slv.indexTo1D(1, sizeY-1, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[slv.indexTo1D(0, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_v[slv.indexTo1D(0, sizeY-2, sizeX)] + new_v[slv.indexTo1D(1, sizeY-1, sizeX)]),0))
    # Upper-right corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[slv.indexTo1D(sizeX-1, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_u[slv.indexTo1D(sizeX-1, sizeY-2, sizeX)] + new_u[slv.indexTo1D(sizeX-2, sizeY-1, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[slv.indexTo1D(sizeX-1, sizeY-1, sizeX)]], 0.5*tf.expand_dims((new_v[slv.indexTo1D(sizeX-1, sizeY-2, sizeX)] + new_v[slv.indexTo1D(sizeX-2, sizeY-1, sizeX)]),0))
    # Bottom-left corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[slv.indexTo1D(0, 0, sizeX)]], 0.5*tf.expand_dims((new_u[slv.indexTo1D(0, 1, sizeX)] + new_u[slv.indexTo1D(1, 0, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[slv.indexTo1D(0, 0, sizeX)]], 0.5*tf.expand_dims((new_v[slv.indexTo1D(0, 1, sizeX)] + new_v[slv.indexTo1D(1, 0, sizeX)]),0))
    # Bottom-right corner
    new_u = tf.tensor_scatter_nd_update(new_u, [[slv.indexTo1D(sizeX-1, 0, sizeX)]], 0.5*tf.expand_dims((new_u[slv.indexTo1D(sizeX-2, 0, sizeX)] + new_u[slv.indexTo1D(sizeX-1, 1, sizeX)]),0))
    new_v = tf.tensor_scatter_nd_update(new_v, [[slv.indexTo1D(sizeX-1, 0, sizeX)]], 0.5*tf.expand_dims((new_v[slv.indexTo1D(sizeX-2, 0, sizeX)] + new_v[slv.indexTo1D(sizeX-1, 1, sizeX)]),0))

    return new_u, new_v

BOUNDARY_FUNC = set_obstacles_boundary
trained_vel_x, trained_vel_y =  train.train(MAX_ITER, density_init, target_density, N_FRAMES, u_init, v_init, FLUID_SETTINGS, COORDS_X, COORDS_Y, BOUNDARY_FUNC, FILENAME, debug=False)
