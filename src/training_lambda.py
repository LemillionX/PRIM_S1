import tensorflow as tf
import tf_solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def loss_quadratic(current, target):
    return 0.5*(tf.norm(current - target)**2)


##############################################################
#               General settings
##############################################################
'''
Target shape
111111
011110
001100
001100
011110
111111
'''
target_density =[
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,   
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,  
 0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,  
 0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,  
 0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,  
 0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,  
 0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

SIZE_X = int(np.sqrt(len(target_density)))          # number of elements in the x-axis
SIZE_Y = int(np.sqrt(len(target_density)))           # number of elements in the y-axis
assert (SIZE_X == SIZE_Y), "Dimensions on axis are different !"
TIMESTEP = 0.025
N_FRAMES = 20     # number of frames to draw
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


boundary_func = None

## Pre-build matrices
laplace_mat =  tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, 1/(D*D), -4/(D*D)), dtype=tf.float32)
velocity_diff_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -VISC*TIMESTEP/(D*D), 1+4*VISC*TIMESTEP/(D*D) ), dtype=tf.float32)
scalar_diffuse_mat = tf.convert_to_tensor(slv.build_laplacian_matrix(SIZE_X, SIZE_Y, -K_DIFF*TIMESTEP/(D*D), 1+4*K_DIFF*TIMESTEP/(D*D) ), dtype=tf.float32)


## Converting variables to tensors
target_density = tf.convert_to_tensor(target_density, dtype=tf.float32)
velocity_field_x = tf.convert_to_tensor(u_init, dtype=tf.float32)
velocity_field_y = tf.convert_to_tensor(v_init, dtype=tf.float32)
density_field = tf.convert_to_tensor(density_init, dtype=tf.float32)
trained_density = tf.convert_to_tensor(density_init, dtype=tf.float32)
trained_vel_x = tf.convert_to_tensor(velocity_field_x, dtype=tf.float32)
trained_vel_y = tf.convert_to_tensor(velocity_field_y, dtype=tf.float32)
dt = tf.convert_to_tensor(TIMESTEP, dtype=tf.float32)
COORDS_X = tf.convert_to_tensor(COORDS_X, dtype=tf.float32)
COORDS_Y = tf.convert_to_tensor(COORDS_Y, dtype=tf.float32)


#############################################################
#                   Training                                #
#############################################################
# Initial guess
loss = loss_quadratic(density_field, target_density)
old_loss = loss +1 
with tf.GradientTape() as tape:
    velocity_field_x = tf.Variable(velocity_field_x)
    velocity_field_y = tf.Variable(velocity_field_y)
    _, _, density_field = slv.simulate(N_FRAMES, velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF, boundary_func, leave=False)
    loss = loss_quadratic(density_field, target_density)
grad = tape.gradient([loss], [velocity_field_x, velocity_field_y])
print("[step 0] : gradient norm = ",tf.norm(grad).numpy())

# Optimisation
count = 0
while (count < 300 and loss > 0.1 and tf.norm(grad).numpy() > 0.01):
    # alpha = 0.01*abs(tf.random.normal([1]))
    old_loss = loss
    alpha = tf.constant(1.1/np.sqrt(count+1,),dtype = tf.float32)
    density_field = tf.convert_to_tensor(density_init, dtype=tf.float32)
    trained_vel_x = trained_vel_x - alpha*grad[0]
    trained_vel_y = trained_vel_y - alpha*grad[1]
    with tf.GradientTape() as tape:
        velocity_field_x = tf.Variable(trained_vel_x)
        velocity_field_y = tf.Variable(trained_vel_y)
        _,_, density_field = slv.simulate(N_FRAMES, velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF, boundary_func, leave=False)
        loss = loss_quadratic(density_field, target_density)
    count += 1
    grad = tape.gradient([loss], [velocity_field_x, velocity_field_y])
    # print(grad)
    if (count < 3) or (count%10 == 0):
        print("[step", count, "] : alpha = ", alpha.numpy(), ", loss = ", loss.numpy(), ", gradient norm = ", tf.norm(grad).numpy())

print("After ", count, " iterations, the velocity field is " )
print("x component = ")
print(trained_vel_x)
print("y component = ")
print(trained_vel_y)
print(" and gradient norm = ",tf.norm(grad).numpy())


#################################################################
#                       Testing                                 #
#################################################################
## Plot initialisation 
x,y = np.meshgrid(COORDS_X[:SIZE_X], COORDS_Y[::SIZE_X])
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
Q = ax.quiver(x, y, viz.tensorToGrid(u_init, SIZE_X, SIZE_Y), viz.tensorToGrid(v_init, SIZE_X, SIZE_Y), color='red', scale_units='width')

density_field = tf.convert_to_tensor(density_init, dtype=tf.float32)
velocity_field_x = trained_vel_x
velocity_field_y = trained_vel_y

## Plot Animation
OUTPUT_DIR = "output"
FOLDER_NAME = "trained_velocity"
DENSITY_NAME = "trained_density"
DIR_PATH = os.path.join(os.getcwd().rsplit("\\",1)[0], OUTPUT_DIR)
SAVE_PATH =  os.path.join(DIR_PATH, FOLDER_NAME)
FPS = 20
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
if not os.path.isdir(os.path.join(DIR_PATH, DENSITY_NAME)):
    os.mkdir(os.path.join(DIR_PATH, DENSITY_NAME))

print(SAVE_PATH)
pbar = tqdm(range(1, N_FRAMES*2+1), desc = "Simulating....")
plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(0)))
viz.draw_density(viz.tensorToGrid(density_field, SIZE_X, SIZE_Y), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(0)))

for t in pbar:
    velocity_field_x, velocity_field_y, density_field = slv.update(velocity_field_x, velocity_field_y, density_field ,SIZE_X, SIZE_Y, COORDS_X, COORDS_Y, dt, GRID_MIN, D, laplace_mat, ALPHA, velocity_diff_mat, VISC, scalar_diffuse_mat, K_DIFF, boundary_func)
    # Viz update
    viz.draw_density(tf.reshape(density_field, shape=(SIZE_X, SIZE_Y)).numpy(), os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    u_viz = tf.reshape(velocity_field_x, shape=(SIZE_X, SIZE_Y)).numpy()
    v_viz = tf.reshape(velocity_field_y, shape=(SIZE_X, SIZE_Y)).numpy()
    Q.set_UVC(u_viz,v_viz)
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)