import sys
sys.path.insert(0, '../src')
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np
import tf_solver_staggered as slv
from tqdm import tqdm
import json
import fluid_layer
import time

class Fluid():
    def __init__(self, layer_size=800):
        
        self.size = 20
        self.u = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.v = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.d = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.layer = fluid_layer.FluidLayer(layer_size, self.size)
        self.layer.timer.timeout.connect(self.update_frame)

        # Training settings
        self.learning_rate = 1.5
        self.weight = 1

        # Simulation settings
        self.Nframes = 100
        self.dt = 0.025
        self.grid_min = -1
        self.grid_max = 1
        self.h = (self.grid_max - self.grid_min)/self.size
        self.diffusion_coeff = 0.0
        self.dissipation_rate = 0.0
        self.viscosity = 0.0
        self.boundary = "dirichlet"
        self.useSource = False
        self.sourceDuration = 0
        self.source = None

        # Intermediate variables
        self.coordsX = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.coordsY = tf.zeros([self.size*self.size], dtype=tf.float32)

        # Simulation folders
        self.filename = "fluid_simulation"
        self.file_to_play = None

        # Initial drawing
        self.drawDensity(self.d)

    def setCoords(self):
        coordsX = np.zeros(self.size*self.size)
        coordsY = np.zeros(self.size*self.size)
        for j in range(self.size):
            for i in range(self.size):
                coordsX[i+j*self.size] = self.grid_min + (i+0.5)*self.h
                coordsY[i+j*self.size] = self.grid_min + (j+0.5)*self.h
        self.coordsX = tf.convert_to_tensor(coordsX, dtype=tf.float32)
        self.coordsY = tf.convert_to_tensor(coordsY, dtype=tf.float32)

    def setSize(self, size):
        self.size = size
        self.u = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.v = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.d = tf.zeros([self.size*self.size], dtype=tf.float32)
        self.h = (self.grid_max - self.grid_min)/self.size
        self.setCoords()
        self.layer.grid_size = size

    def setDensity(self, density):
        self.d = tf.convert_to_tensor(density, dtype=tf.float32)

    def setSource(self):
        if self.useSource:
            self.source = {}
            self.source["value"] = 1.0
            indices = np.where(self.d.numpy() == 1.0)[0]
            self.source["indices"] = indices.reshape((np.shape(indices)[0], 1))
            self.source["time"] = self.sourceDuration
        else:
            self.source = None


    def buildMatrices(self):
        lu, p = slv.build_laplacian_matrix(self.size, self.size, 1/(self.h*self.h), -4/(self.h*self.h), self.boundary)
        velocity_diff_LU, velocity_diff_P = slv.build_laplacian_matrix(self.size, self.size, -self.viscosity*self.dt/(self.h*self.h), 1+4*self.viscosity*self.dt/(self.h*self.h))
        scalar_diffuse_LU, scalar_diffuse_P = slv.build_laplacian_matrix(self.size, self.size, -self.diffusion_coeff*self.dt/(self.h*self.h), 1+4*self.diffusion_coeff*self.dt/(self.h*self.h) )
        return lu, p, velocity_diff_LU, velocity_diff_P, scalar_diffuse_LU, scalar_diffuse_P

    def bakeSimulation(self, lu, p, velocity_diff_LU, velocity_diff_P, scalar_diffuse_LU, scalar_diffuse_P, file):
        # Intermediate variables
        u = tf.identity(self.u)
        v = tf.identity(self.v)
        d = tf.identity(self.d)
        simulated_u = []
        simulated_v = []         
        simulated_d = []         

        self.setSource()

        pbar = tqdm(range(1, self.Nframes+1), desc="Baking simulation...")
        for t in pbar:
            f_u, f_v = slv.buoyancyForce(d, self.size, self.size, self.coordsX, self.coordsY, self.grid_min, self.h)
            u,v,d = slv.update(u, v, d, self.size, self.size, self.coordsX, self.coordsY, self.dt, self.grid_min, self.h, lu, p,
                               self.dissipation_rate, velocity_diff_LU, velocity_diff_P, self.viscosity, scalar_diffuse_LU, scalar_diffuse_P, self.diffusion_coeff,
                               boundary_func=self.boundary, source=self.source, t=t, f_u=f_u, f_v=f_v)
            simulated_u.append(u.numpy().tolist())
            simulated_v.append(v.numpy().tolist())
            simulated_d.append(d.numpy().tolist())

        with open(file, 'w') as f:
            if self.source is not None:
                self.source["indices"] = self.source["indices"].tolist()
            json.dump({
                "GRID_SIZE":self.size,
                "N_FRAMES": self.Nframes,
                "TIMESTEP": self.dt,
                "GRID_MIN": self.grid_min,
                "GRID_MAX": self.grid_max,
                "h": self.h,
                "DIFFUSION_COEFF": self.diffusion_coeff,
                "DISSIPATION_RATE": self.dissipation_rate,
                "VISCOSITY": self.viscosity,
                "BOUNDARY": self.boundary,
                "SOURCE": self.source,
                "SOURCE_DURATION": self.sourceDuration,
                "LEARNING_RATE": self.learning_rate,
                "WEIGHT": self.weight,
                "u": simulated_u,
                "v": simulated_v,
                "d": simulated_d
            },
            f, indent=4)
            print(file)

    def drawDensity(self, density):
        r= 255
        g= 0
        b= 0
        blocSize = int(self.layer.size/self.size)
        self.layer.clean()
        for i in range(self.size):
            for j in range(self.size):
                self.layer.drawCell(blocSize, self.size, i,j, r,g,b,int(255*density[i+j*self.size]))

    def loadSettings(self, settings):
        self.Nframes = settings["N_FRAMES"]
        self.dt = settings["TIMESTEP"]
        self.grid_min = settings["GRID_MIN"]
        self.grid_max = settings["GRID_MAX"]
        self.h = settings["h"]
        self.diffusion_coeff = settings["DIFFUSION_COEFF"]
        self.dissipation_rate = settings["DISSIPATION_RATE"]
        self.viscosity = settings["VISCOSITY"]
        self.boundary = settings["BOUNDARY"]
        self.sourceDuration = settings["SOURCE_DURATION"]
        self.source = settings["SOURCE"]
        self.learning_rate = settings["LEARNING_RATE"]
        self.weight = settings["WEIGHT"]

    def playDensity(self, dir_path):
        if self.file_to_play is not None:
            with open(dir_path+self.file_to_play+".json", 'r') as file:
                data = json.load(file)
                self.loadSettings(data)
                self.layer.densities = data["d"]
            self.layer.update()
            self.layer.timer.start(self.layer.interval)

    def update_frame(self):
        self.layer.current_frame += 1
        if self.layer.current_frame+1 >= len(self.layer.densities):
            self.layer.current_frame = 0
            self.layer.timer.stop()
            print("Done !")
        self.layer.update()
