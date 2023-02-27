import solver as slv
import numpy as np
import viz as viz
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from tqdm import tqdm


##############################################################
#               General settings
##############################################################

N_DIM = 2
GRID_MIN = -1
GRID_MAX = 1
ORIGIN = np.array([GRID_MIN, GRID_MIN])
NB_CELLS = 99 #number of cells on one dimension of the grid
L = GRID_MAX - GRID_MIN
N = np.array([NB_CELLS for i in range(N_DIM)])
D = L/NB_CELLS
VISC = 0.0
N_ITER = 50
N_FRAMES = 300

grid_x = np.array([ORIGIN[0] + (i+0.5)*D for i in range(N[0])])
grid_y = np.array([ORIGIN[1] + (j+0.5)*D for j in range(N[1])])

DEBUG = False
dt = 0.025

#################################################################
# Initialisation
#################################################################
COORDS = np.zeros((N[0], N[1], N_DIM))


# Setup coordinates grid
for i in range(N[0]):
    for j in range(N[1]):
        x = ORIGIN + np.array([i+0.5, j+0.5])*D 
        COORDS[j,i] = x

# Setup velocity field and scalar field

## Velocity field settings
CENTER = ORIGIN + np.array([int((NB_CELLS/2)) + 0.5, int((NB_CELLS/2)) + 0.5])*D
VORTEX_RADIUS = 0.5*L/2.0
u_init = np.zeros((N[0], N[1], N_DIM))

## Scalar field settings
SOURCE = 0
k_diff = 0.0      #diffusion constant
alpha = 0.0       #dissipation rate
density_field = np.zeros((N[0], N[1]))

for i in range(COORDS.shape[0]):
    for j in range(COORDS.shape[1]):
        point = COORDS[i,j] - CENTER
        r = np.linalg.norm(point)
        if r < VORTEX_RADIUS:
            u_init[i,j] = 5*np.array([point[1], -point[0]])
            density_field[i,j] = 1.0
        # u_init[i,j] = np.array([ np.sin(2*np.pi*COORDS[i,j,1]), np.sin(2*np.pi*COORDS[i,j,0])])
        # u_init[i,j] = D*np.array([ 1 , np.sin(2*np.pi*COORDS[i,j,0])])


for d in range(N_DIM):
    u_init[:,:,d] = slv.set_solid_boundary(u_init[:,:,d])
density_field = slv.set_solid_boundary(density_field)
velocity_field = np.copy(u_init)

##############################################################
#               Plot initialisation 
##############################################################
RESOLUTION = 25
GRID_STEP = 1
if RESOLUTION < NB_CELLS:
    GRID_STEP = NB_CELLS // RESOLUTION + 1

x,y = np.meshgrid(grid_x[::GRID_STEP], grid_y[::GRID_STEP])
ux_init = u_init[::GRID_STEP,::GRID_STEP,0]
uy_init = u_init[::GRID_STEP,::GRID_STEP,1]
fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal', adjustable='box')
Q = ax.quiver(x, y, ux_init, uy_init, color='red', scale_units='width')

##############################################################
#               Solving
##############################################################
def simulate(t, u1, s0):
    u0 =  np.copy(u1)
    ## Vstep
    # advection step
    for i in range(N_DIM):
        u1[:,:,i] = slv.advect(u0[:,:,i], u0, dt, COORDS, ORIGIN, D, GRID_MAX, GRID_MIN, True)
    u0 = np.copy(u1)

    # diffusion step
    for i in range(N_DIM):
        u1[:,:,i] = slv.diffuse(u1[:,:,i], u0[:,:,i], VISC, dt, D, N_ITER)
    u0 = np.copy(u1)

    # projection step
    u1 = slv.project(u1, u0, D, N_ITER)

    u = u1[::GRID_STEP,::GRID_STEP,0]
    v = u1[::GRID_STEP,::GRID_STEP,1]

    ## Sstep
    s_temp = np.copy(s0)
    s0 = slv.advect(s0, u1, dt, COORDS, ORIGIN, D, GRID_MAX, GRID_MIN)
    s_temp = np.copy(s0)
    s0 = slv.diffuse(s0, s_temp, k_diff, dt, D, N_ITER)
    s0 = slv.dissipate(s0, alpha, dt)
    Q.set_UVC(u,v)
    return u1, s0
##############################################################
#               Real-time Plotting
##############################################################
# ani = animation.FuncAnimation(fig, simulate, fargs=(velocity_field, density_field), interval=dt*1000, blit=False)
# plt.show()


##############################################################
#               Plot Animation
#############################################################
OUTPUT_DIR = "output"
FOLDER_NAME = "gauss_seidel_vel_"+str(NB_CELLS)
DENSITY_NAME = "gauss_seidel_dens_"+str(NB_CELLS)
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
    velocity_field, density_field = simulate(t, velocity_field, density_field)
    viz.draw_density(density_field, os.path.join(DIR_PATH, DENSITY_NAME, '{:04d}.png'.format(t)))
    plt.savefig(os.path.join(SAVE_PATH, '{:04d}'.format(t)))

viz.frames2gif(os.path.join(DIR_PATH, FOLDER_NAME), os.path.join(DIR_PATH, FOLDER_NAME+".gif"), FPS)
viz.frames2gif(os.path.join(DIR_PATH, DENSITY_NAME), os.path.join(DIR_PATH, DENSITY_NAME+".gif"), FPS)

