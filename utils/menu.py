import sys
sys.path.insert(0, '../src')
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QLocale
import callbacksUI as callback
import canvas
import numpy as np
import fluid
import tensorflow as tf
import tf_solver_staggered as slv
import tf_train as train

class Menu(QtWidgets.QVBoxLayout):

    def __init__(self, window):
        super().__init__()

        self.window = window
        self.canvas = canvas.Canvas()
        self.canvas.mode = ""
        self.fluid = fluid.Fluid(layer_size=self.canvas.size)

        # General Widgets
        self.generalWidgets = QtWidgets.QFormLayout()

        self.resolutionText = QtWidgets.QLineEdit(str(self.canvas.gridResolution))
        self.resolutionText.setValidator(QtGui.QIntValidator(10, 128))
        self.resolutionText.setMaxLength(3)
        self.resolutionText.textChanged.connect(self.setResolution)
        self.generalWidgets.addRow("Grid resolution ", self.resolutionText)

        self.drawGridButton = QtWidgets.QCheckBox()
        self.drawGridButton.stateChanged.connect(self.toggleGrid)
        self.generalWidgets.addRow("Draw Grid", self.drawGridButton)

        self.frames = QtWidgets.QLineEdit(str(self.fluid.Nframes))
        self.frames.setValidator(QtGui.QIntValidator())
        self.frames.setMaxLength(4)
        self.frames.textChanged.connect(self.setFrames)
        self.generalWidgets.addRow("Frames", self.frames)

        double_validator = QtGui.QDoubleValidator(0.0001, 10.0000, 4, notation=QtGui.QDoubleValidator.StandardNotation)
        double_validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.dt = QtWidgets.QLineEdit(str(self.fluid.dt))
        self.dt.setValidator(double_validator)
        self.dt.textChanged.connect(self.setTimestep)
        self.generalWidgets.addRow("Timestep", self.dt)

        self.boundary = QtWidgets.QComboBox()
        self.boundary.addItems([ 'dirichlet', 'neumann'])
        self.boundary.currentTextChanged.connect(self.setBoundary)
        self.generalWidgets.addRow("Boundary Conditions", self.boundary)

        self.source = QtWidgets.QCheckBox('Add Source')
        self.source.stateChanged.connect(self.setSource)
        self.generalWidgets.addWidget(self.source)
        self.sourceDuration = QtWidgets.QLineEdit(str(self.fluid.sourceDuration))
        self.sourceDuration.setValidator(QtGui.QIntValidator())
        self.sourceDuration.setMaxLength(4)
        self.sourceDuration.textChanged.connect(self.setSourceDuration)
        self.generalWidgets.addRow("Source Frames", self.sourceDuration)

        self.bakeFile = QtWidgets.QLineEdit(self.fluid.filename)
        self.bakeFile.textChanged.connect(self.setBakeFile)
        self.generalWidgets.addRow("File to bake", self.bakeFile)

        self.addLayout(self.generalWidgets)

        # Layout Chooser
        self.layoutChooser = QtWidgets.QComboBox()
        self.layoutChooser.addItems(['fluid_simulation', 'fluid_constraints'])
        self.generalWidgets.addRow("Mode", self.layoutChooser)
        self.layoutChooser.currentIndexChanged.connect(self.setLayout)

        # Different layouts
        self.fluid_constraint_container = QtWidgets.QWidget()
        self.fluid_simulation_container = QtWidgets.QWidget()
        self.fluidConstraintsLayout = QtWidgets.QFormLayout(self.fluid_constraint_container)
        self.fluidSimulationLayout = QtWidgets.QFormLayout(self.fluid_simulation_container)

        # Fluid Constraints Layout
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(['trajectory', 'initial_density', 'target_density'])
        self.fluidConstraintsLayout.addRow("Edit :", self.combobox)
        self.combobox.currentTextChanged.connect(self.setMode)

        self.saveButton = QtWidgets.QPushButton('Save Config')
        self.saveButton.clicked.connect(self.save_config)
        self.fluidConstraintsLayout.addWidget(self.saveButton)

        self.loadButton = QtWidgets.QPushButton('Load Config')
        self.loadButton.clicked.connect(self.load_config)
        self.fluidConstraintsLayout.addWidget(self.loadButton)

        self.resetTrajButton = QtWidgets.QPushButton('Reset Trajectory')
        self.resetTrajButton.clicked.connect(self.reset_config)
        self.fluidConstraintsLayout.addWidget(self.resetTrajButton)

        self.max_iter = QtWidgets.QLineEdit(str(self.fluid.max_iter))
        self.max_iter.setValidator(QtGui.QIntValidator())
        self.max_iter.setMaxLength(3)
        self.max_iter.textChanged.connect(self.setMaxIter)
        self.fluidConstraintsLayout.addRow("Max N_Iter", self.max_iter)

        self.learningRateButton = QtWidgets.QLineEdit(str(self.fluid.learning_rate))
        self.learningRateButton.textChanged.connect(self.setLearningRate)
        self.learningRateButton.setValidator(double_validator)
        self.fluidConstraintsLayout.addRow("Learning rate", self.learningRateButton)

        self.weightButton = QtWidgets.QLineEdit(str(self.fluid.weight))
        self.weightButton.textChanged.connect(self.setWeight)
        self.weightButton.setValidator(double_validator)
        self.fluidConstraintsLayout.addRow("Trajectory weight", self.weightButton)

        self.trainButton = QtWidgets.QPushButton('Train')
        self.trainButton.clicked.connect(self.train)
        self.fluidConstraintsLayout.addWidget(self.trainButton)


        # Fluid Settings Layout
        self.bakeButton = QtWidgets.QPushButton('Bake Simulation')
        self.bakeButton.clicked.connect(self.bake)
        self.fluidSimulationLayout.addWidget(self.bakeButton)

        self.playFile = QtWidgets.QLineEdit()
        self.playFile.textChanged.connect(self.setFileToPlay)
        self.fluidSimulationLayout.addRow("File to play", self.playFile)     

        self.colorButton = QtWidgets.QPushButton('Change Smoke color')
        self.colorButton.clicked.connect(self.setDensityColor)
        self.fluidSimulationLayout.addWidget(self.colorButton)

        self.playButton = QtWidgets.QPushButton('Play')
        self.playButton.clicked.connect(self.play)
        self.fluidSimulationLayout.addWidget(self.playButton)


        # Stack Layouts
        self.stacked_layout = QtWidgets.QStackedLayout()
        self.stacked_layout.addWidget(self.fluid_simulation_container)
        self.stacked_layout.addWidget(self.fluid_constraint_container)
        self.addLayout(self.stacked_layout)

    def setLayout(self, index):
        self.stacked_layout.setCurrentIndex(index)
        self.window.stacked_layout.setCurrentIndex(index)
        if index == 0:
            self.fluid.setSize(self.canvas.gridResolution)
            self.fluid.d = tf.convert_to_tensor(self.canvas.initialDensity, dtype=tf.float32)
            self.fluid.layer.densities = [self.fluid.d.numpy()]
        
        self.fluid.layer.update()
        self.combobox.setCurrentText("trajectory")
        self.canvas.mode = "trajectory"

    def setBakeFile(self, filename):
        self.fluid.filename = filename

    def setFileToPlay(self, text):
        if len(text.strip()) > 0:
            self.fluid.file_to_play = text
        else:
            self.fluid.file_to_play = None

    def setMaxIter(self, value):
        if len(value.strip()) > 0:
            self.fluid.max_iter = int(value)
        else:
            self.fluid.max_iter = 0

    def setLearningRate(self, value):
        if len(value.strip()) > 0:
            self.fluid.learning_rate = float(value)
        else:
            self.fluid.learning_rate = 1.0000

    def setWeight(self, value):
        if len(value.strip()) > 0:
            self.fluid.weight = float(value)
        else:
            self.fluid.weight = 1.0000

    def setResolution(self, text):
        if len(text.strip()) > 0:
            resolution= int(text)
        else:
            resolution = 10
        self.canvas.setGridResolution(resolution, self.drawGridButton.isChecked())
        self.fluid.setSize(resolution)

    def setDensityColor(self):
        color_dialog = QtWidgets.QColorDialog()
        color_dialog.currentColorChanged.connect(self.fluid.layer.setRGB)
        color_dialog.exec()
        # color = QtWidgets.QColorDialog.getColor()
        # if color.isValid():
        #     self.fluid.layer.r, self.fluid.layer.g, self.fluid.layer.b, _ = color.getRgb()
        # self.fluid.layer.update()

    def setFrames(self, frames):
        if len(frames.strip()) > 0:
            self.fluid.Nframes = int(frames)
        else:
            self.fluid.Nframes = 0

    def setTimestep(self, dt):
        if len(dt.strip()) > 0:
            self.fluid.dt = float(dt)
        else:
            self.fluid.dt = 0.001

    def setBoundary(self, text):
        self.fluid.boundary = text    

    def setSource(self, state):
        if state == QtCore.Qt.Checked:
            self.fluid.useSource = True
        else:
            self.fluid.useSource = False
    
    def setSourceDuration(self, frame):
        if len(frame.strip()) > 0:
            self.fluid.sourceDuration = int(frame)
        else:
            self.fluid.sourceDuration = 1

    def setCanvas(self, canva):
        self.canvas = canva

    def setMode(self, mode):
        print("Entering mode : ", mode)
        self.canvas.mode = mode

    def save_config(self):
        print("Saving file")
        file_name = callback.prompt_file()
        if file_name is not None:
            indices = callback.points2indices(self.canvas.curves, int(self.canvas.blocSize), self.canvas.gridResolution)
            callback.saveToJSON(indices[0], self.canvas.targetDensity.tolist(), self.canvas.initialDensity.tolist(), self.canvas.curves, self.canvas.gridResolution, file_name)
        print("Trajectory saved here : ", file_name)

    def load_config(self):
        print("Load file")
        data = callback.loadFromJSON()
        if data is not None:
            self.canvas.clean()
            self.canvas.setGridResolution(int(np.sqrt(len(data["init_density"]))))
            self.fluid.setSize(int(np.sqrt(len(data["init_density"]))))
            self.canvas.hideGrid()
            if self.drawGridButton.isChecked():
                self.canvas.drawGrid()
            self.resolutionText.setText(str(self.canvas.gridResolution))

            self.canvas.setInitialDensity(data["init_density"])
            self.canvas.setTargetDensity(data["target_density"])
            self.fluid.setDensity(data["init_density"])
            if "curves" in data:
                self.canvas.setCurves(data["curves"])

    def reset_config(self):
        print("Reset trajectory")
        self.canvas.clean()
        self.canvas.setInitialDensity(self.canvas.initialDensity)
        self.canvas.setTargetDensity(self.canvas.targetDensity)

    def updateSettingsFields(self):
        self.resolutionText.setText(str(self.fluid.size))
        self.frames.setText(str(self.fluid.Nframes))
        self.dt.setText(str(self.fluid.dt))
        self.boundary.setCurrentText(self.fluid.boundary)
        self.sourceDuration.setText(str(self.fluid.sourceDuration))

    def toggleGrid(self, state):
        if state == QtCore.Qt.Checked:
            self.canvas.drawGrid()
            self.fluid.layer.drawGrid = True
        else:
            self.canvas.hideGrid()
            self.fluid.layer.drawGrid = False
        self.fluid.layer.update()

    def bake(self):
        print("Baking Simulation")
        lu, p, velocity_diff_LU, velocity_diff_P, scalar_diffuse_LU, scalar_diffuse_P = self.fluid.buildMatrices()
        file = "../bake/{filename}.json".format(filename=self.fluid.filename)
        self.fluid.bakeSimulation(lu, p, velocity_diff_LU, velocity_diff_P, scalar_diffuse_LU, scalar_diffuse_P, file)
        self.fluid.file_to_play = self.fluid.filename
        self.playFile.setText(self.fluid.file_to_play)
        print("Simulation baked !")

    def play(self):
        print("Play Simulation...")
        self.fluid.playDensity("../bake/")
        self.updateSettingsFields()

    def train(self):
        print("Optimizing under constraints...")
        u_init = np.zeros(self.fluid.size*self.fluid.size)
        v_init = np.zeros(self.fluid.size*self.fluid.size)
        constraints = {}
        cells = callback.points2indices(self.canvas.curves, int(self.canvas.blocSize), self.canvas.gridResolution)[0]
        constraints["values"] = []
        constraints["indices"] = []
        for i in range(1, len(cells)-1):
            constraints["indices"].append([cells[i][0] + cells[i][1] * self.fluid.size ])
            constraints["values"].append([[cells[i+1][0] - cells[i-1][0]], [cells[i+1][1] - cells[i-1][1]]])

        if len(constraints["indices"]) > 0:
            print("Velocity is constrained")
            constraints["values"] = np.array(constraints["values"])
            idx = np.array(constraints["indices"]).flatten()
            u_init[idx] = constraints["values"][:, :, 0][:, 0]
            v_init[idx] = constraints["values"][:, :, 0][:, 1]
            constraints["weights"] = tf.convert_to_tensor(self.fluid.weight*np.ones(len(constraints["indices"])), dtype=tf.float32)
            constraints["keyframes"] = [round((i+1)*self.fluid.Nframes/(len(constraints["indices"])+1)) for i in range(len(constraints["indices"]))]
        else:
            constraints = None


        trained_u, trained_v = train.trainUI(self.fluid.max_iter, self.canvas.initialDensity, self.canvas.targetDensity,
                                             self.fluid.Nframes, u_init, v_init, self.fluid.__dict__, self.fluid.coordsX,
                                             self.fluid.coordsY, constraints, self.fluid.learning_rate)
        print("Optimisation process done")
        self.layoutChooser.setCurrentText("fluid_simulation")
        self.setLayout(0)
        self.fluid.u = trained_u
        self.fluid.v = trained_v
        self.bake()
