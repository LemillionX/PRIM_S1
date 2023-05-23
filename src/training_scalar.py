import tf_train as train
import numpy as np
import json

# File settings
FILENAME = {}
FILENAME["velocity"] = "velocity"
FILENAME["density"] = "density"

# Simulation settings
MAX_ITER = 50
LEARNING_RATE = 1.0
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


# Load data from .json file
CONSTRAINT = {}
CONSTRAINT_FILE = "batch1_traj3"
with open("../data/"+CONSTRAINT_FILE+".json") as file:
    print('Loading file', CONSTRAINT_FILE+".json")
    CONSTRAINT = json.load(file)

# Data always have to contain target and initial density
target_density = CONSTRAINT["target_density"][:]
density_init = CONSTRAINT["init_density"][:]
SIZE = int(np.sqrt(len(target_density)))

# Source settings
# FLUID_SETTINGS["source"] = {}
# FLUID_SETTINGS["source"]["value"]=1.0
# src_indices = np.where(np.array(density_init)==1.0)[0] 
# FLUID_SETTINGS["source"]["indices"]=src_indices.reshape((src_indices.shape[0],1)) 
# FLUID_SETTINGS["source"]["time"]=20


# Calculate some useful physicial quantities
D = (FLUID_SETTINGS["grid_max"] -FLUID_SETTINGS["grid_min"])/SIZE
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
a_init = np.random.rand(SIZE * SIZE)

for j in range(SIZE):
    for i in range(SIZE):
        point_x = FLUID_SETTINGS["grid_min"]+(i+0.5)*D
        point_y = FLUID_SETTINGS["grid_min"]+(j+0.5)*D
        COORDS_X.append(point_x)
        COORDS_Y.append(point_y)

if len(CONSTRAINT["indices"]) > 0:
    print("Velocity is constrained")
    # Check if there is a trajectory constraint
    CONSTRAINT["values"] = np.array(CONSTRAINT["values"])
    CONSTRAINT["keyframes"] = [round((i+1)*N_FRAMES/(len(CONSTRAINT["indices"])+1)) for i in range(len(CONSTRAINT["indices"]))]
    CONSTRAINT["weights"] = [WEIGHT  for _ in range(len(CONSTRAINT["indices"]))]
else:
    print("Velocity is NOT constrained")
    CONSTRAINT = None

BOUNDARY_FUNC = None

dt = FLUID_SETTINGS["timestep"]
trained_a = train.train_scalar_field(MAX_ITER, density_init, target_density, N_FRAMES, a_init, FLUID_SETTINGS, COORDS_X, COORDS_Y, BOUNDARY_FUNC, FILENAME, CONSTRAINT, LEARNING_RATE, debug=False)

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
               "trained_a": trained_a.numpy().tolist()
               },
               file, indent=4)
    

