import tf_train as train
import numpy as np

FILENAME = {}
FILENAME["velocity"] = "trained_velocity_lambda"
FILENAME["density"] = "trained_density_lambda"

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

MAX_ITER = 200
N_FRAMES = 20     # number of the frame where we want the shape to be matched
FLUID_SETTINGS = {}
FLUID_SETTINGS["timestep"] = 0.025
FLUID_SETTINGS["grid_min"] = -1
FLUID_SETTINGS["grid_max"] = 1
FLUID_SETTINGS["diffusion_coeff"] = 0.0
FLUID_SETTINGS["dissipation_rate"] = 0.0
FLUID_SETTINGS["viscosity"] = 0.0
FLUID_SETTINGS["source"] = None

SIZE = int(np.sqrt(len(target_density)))
D = (FLUID_SETTINGS["grid_max"] -FLUID_SETTINGS["grid_min"])/SIZE

VORTEX_RADIUS = 0.5*(FLUID_SETTINGS["grid_max"] - FLUID_SETTINGS["grid_min"])/2.0
VORTEX_CENTER = np.array([FLUID_SETTINGS["grid_min"] + D*(int(SIZE/2) + 0.5), FLUID_SETTINGS["grid_min"] + D*(int(SIZE/2)+0.5)])
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
u_init = []     # x-coordinates of speed
v_init = []     # y-coordinates of speed
density_init = []    # density field
for j in range(SIZE):
    for i in range(SIZE):
        point_x = FLUID_SETTINGS["grid_min"] +(i+0.5)*D
        point_y = FLUID_SETTINGS["grid_min"] +(j+0.5)*D
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


BOUNDARY_FUNC = None
trained_vel_x, trained_vel_y =  train.train(MAX_ITER, density_init, target_density, N_FRAMES, u_init, v_init, FLUID_SETTINGS, COORDS_X, COORDS_Y, BOUNDARY_FUNC, FILENAME, debug=False)
