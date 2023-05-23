import tf_train as train
import tensorflow as tf
import numpy as np
import json

# File settings
FILENAME = {}
FILENAME["velocity"] = "velocity"
FILENAME["density"] = "density"

# Simulation settings
MAX_ITER = 50
NB_VORTICES = 4
LEARNING_RATE = .01
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

# Calculate some useful physicial quantities
D = (FLUID_SETTINGS["grid_max"] -FLUID_SETTINGS["grid_min"])/SIZE
COORDS_X = []   # x-coordinates of position
COORDS_Y = []   # y-coordinates of position
centers_init = tf.random.uniform([NB_VORTICES,2], minval=FLUID_SETTINGS["grid_min"], maxval=FLUID_SETTINGS["grid_max"])
radius_init = tf.random.uniform([NB_VORTICES])
w_init = tf.random.normal([NB_VORTICES])

centers_init = tf.convert_to_tensor([[0.5, -1.2], [-0.3, -0.5], [0.1, 0], [-0.4, 0.4] ]  , dtype=tf.float32)
radius_init = 0.4*tf.convert_to_tensor([1, 1, 1, 1]  , dtype=tf.float32)
w_init = 30*tf.convert_to_tensor([-1, 1, -0.5, 3.5]  , dtype=tf.float32)

radius_init = tf.Variable(0.4*tf.ones([NB_VORTICES], dtype=tf.float32)) 
w_init = tf.Variable(0.1*tf.convert_to_tensor([1, 1, -2, 2]  , dtype=tf.float32))


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
trained_centers, trained_radius, trained_w = train.train_vortices(MAX_ITER, density_init, target_density, N_FRAMES, NB_VORTICES, centers_init, radius_init, w_init, FLUID_SETTINGS, COORDS_X, COORDS_Y, BOUNDARY_FUNC, FILENAME, CONSTRAINT, LEARNING_RATE, debug=False)
print("trainded_centers = ", trained_centers)

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
               "NB_VORTICES": NB_VORTICES,
               "DIFFUSION_COEFF": FLUID_SETTINGS["diffusion_coeff"],
               "DISSIPATION_RATE": FLUID_SETTINGS["dissipation_rate"],
               "VISCOSITY": FLUID_SETTINGS["viscosity"],
               "SOURCE": FLUID_SETTINGS["source"],
               "trained_centers": trained_centers.numpy().tolist(),
               "trained_radius": trained_radius.numpy().tolist(),
               "trained_w": trained_w.numpy().tolist()
               },
               file, indent=4)
    

