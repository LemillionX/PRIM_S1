import tf_train as train
import numpy as np
import json

# File settings
FILENAME = {}
FILENAME["velocity"] = "trained_velocity_multiple_constraints"
FILENAME["density"] = "trained_density_multiple_constraints"

# Simulation settings
MAX_ITER = 50
LEARNING_RATE = 10
N_FRAMES = 80     # number of the frame where we want the shape to be matched
FLUID_SETTINGS = {}
FLUID_SETTINGS["timestep"] = 0.025
FLUID_SETTINGS["grid_min"] = -1
FLUID_SETTINGS["grid_max"] = 1
FLUID_SETTINGS["diffusion_coeff"] = 0.0
FLUID_SETTINGS["dissipation_rate"] = 0.0
FLUID_SETTINGS["viscosity"] = 0.0
FLUID_SETTINGS["source"] = None

# Load data from .json file
CONSTRAINT = {}
CONSTRAINT_FILE = "test_density.json"
with open("../data/"+CONSTRAINT_FILE) as file:
    print('Loading file', CONSTRAINT_FILE)
    CONSTRAINT = json.load(file)

# Data always have to contain target and initial density
target_density = CONSTRAINT["target_density"][:]
density_init = CONSTRAINT["init_density"][:]
SIZE = int(np.sqrt(len(target_density)))

# Calculate some useful physicial quantities
D = (FLUID_SETTINGS["grid_max"] -FLUID_SETTINGS["grid_min"])/SIZE
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
u_init = []     # x-coordinates of speed
v_init = []     # y-coordinates of speed

for j in range(SIZE):
    for i in range(SIZE):
        point_x = FLUID_SETTINGS["grid_min"]+(i+0.5)*D
        point_y = FLUID_SETTINGS["grid_min"]+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)
        u_init.append(-1)
        v_init.append(1)
         
if len(CONSTRAINT["indices"]) > 0:
    # Check if there is a trajectory constraint
    CONSTRAINT["values"] = (D*np.array(CONSTRAINT["values"]))
    CONSTRAINT["keyframes"] = [round((i+1)*N_FRAMES/(len(CONSTRAINT["indices"])+1)) for i in range(len(CONSTRAINT["indices"]))]
    CONSTRAINT["weights"] = [1  for _ in range(len(CONSTRAINT["indices"]))]
else:
    CONSTRAINT = None

BOUNDARY_FUNC = None
trained_vel_x, trained_vel_y =  train.train(MAX_ITER, density_init, target_density, N_FRAMES, u_init, v_init, FLUID_SETTINGS, COORDS_X, COORDS_Y, BOUNDARY_FUNC, FILENAME, CONSTRAINT, LEARNING_RATE, debug=False)

