import tf_train as train
import tensorflow as tf
import numpy as np
import json
import os

# File settings
FILENAME = {}
FILENAME["velocity"] = "velocity"
FILENAME["density"] = "density"

# Simulation settings
MAX_ITER = 150
LEARNING_RATE = 1
WEIGHT = 1
N_FRAMES = 80     # number of the frame where we want the shape to be matched
FLUID_SETTINGS = {}
FLUID_SETTINGS["timestep"] = 0.025
FLUID_SETTINGS["grid_min"] = -1
FLUID_SETTINGS["grid_max"] = 1
FLUID_SETTINGS["diffusion_coeff"] = 0.0
FLUID_SETTINGS["dissipation_rate"] = 0.0
FLUID_SETTINGS["viscosity"] = 0.0
FLUID_SETTINGS["source"] = None
# FLUID_SETTINGS["source"] = {}
# FLUID_SETTINGS["source"]["value"]=1.0
# FLUID_SETTINGS["source"]["indices"]=np.array([[55],[54],[53],[52],[72],[92],[112],[113],[114],[115],[95],[75],[74],[73],[93],[94]])
# FLUID_SETTINGS["source"]["time"]=20

# Load data from .json file
CONSTRAINT = {}
CONSTRAINT_FILE = "snake"
with open("../data/"+CONSTRAINT_FILE+".json") as file:
    print('Loading file', CONSTRAINT_FILE+".json")
    CONSTRAINT = json.load(file)

# Data always have to contain target and initial density
target_density = CONSTRAINT["target_density"][:]
density_init = CONSTRAINT["init_density"][:]
SIZE = int(np.sqrt(len(target_density)))

# CONSTRAINT = None

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
        u_init.append(0.0)
        v_init.append(0.0)
         
if len(CONSTRAINT["indices"]) > 0:
    print("Velocity is constrained")
    # Check if there is a trajectory constraint
    CONSTRAINT["values"] = np.array(CONSTRAINT["values"])
    u_init = np.zeros_like(u_init)
    v_init = np.zeros_like(v_init)
    idx = np.array(CONSTRAINT["indices"]).flatten()
    u_init[idx] = D*CONSTRAINT["values"][:, :, 0][:, 0]
    v_init[idx] = D*CONSTRAINT["values"][:, :, 0][:, 1]
    # Dummy init
    # u_init = [CONSTRAINT["values"][0][0][0] for _ in range(len(COORDS_X))]
    # v_init = [CONSTRAINT["values"][0][1][0] for _ in range(len(COORDS_Y))]
    CONSTRAINT["keyframes"] = [round((i+1)*N_FRAMES/(len(CONSTRAINT["indices"])+1)) for i in range(len(CONSTRAINT["indices"]))]
    CONSTRAINT["weights"] = [WEIGHT  for _ in range(len(CONSTRAINT["indices"]))]
else:
    print("Velocity is NOT constrained")
    CONSTRAINT = None

BOUNDARY_FUNC = None

dt = FLUID_SETTINGS["timestep"]
trained_vel_x, trained_vel_y =  train.train(MAX_ITER, density_init, target_density, N_FRAMES, u_init, v_init, FLUID_SETTINGS, COORDS_X, COORDS_Y, BOUNDARY_FUNC, FILENAME, CONSTRAINT, LEARNING_RATE, debug=False)


with open("../output/config.json", 'w') as file:
    if FLUID_SETTINGS["source"] is not None:
        FLUID_SETTINGS["source"]["indices"] = FLUID_SETTINGS["source"]["indices"].tolist()
    json.dump({"MAX_ITER": MAX_ITER,
               "LEARNING_RATE": LEARNING_RATE,
               "WEIGHT": WEIGHT,
               "N_FRAMES": N_FRAMES,
               "TIMESTEP": str(dt),
               "GRID_MIN": FLUID_SETTINGS["grid_min"],
               "GRID_MAX": FLUID_SETTINGS["grid_max"],
               "DIFFUSION_COEFF": FLUID_SETTINGS["diffusion_coeff"],
               "DISSIPATION_RATE": FLUID_SETTINGS["dissipation_rate"],
               "VISCOSITY": FLUID_SETTINGS["viscosity"],
               "SOURCE": FLUID_SETTINGS["source"],
               "trained_u": trained_vel_x.numpy().tolist(),
               "trained_v": trained_vel_y.numpy().tolist()
               },
               file, indent=4)
    

