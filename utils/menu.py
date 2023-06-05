from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QLocale
import callbacksUI as callback
import canvas
import numpy as np
import fluid
import tensorflow as tf

class Menu(QtWidgets.QVBoxLayout):

    def __init__(self, window):
        super().__init__()

        self.window = window
        self.canvas = canvas.Canvas()
        self.canvas.mode = ""
        self.fluid = fluid.Fluid(layer_size=self.canvas.size)

        # General Widgets
        self.resolutionText = QtWidgets.QLabel()
        self.resolutionText.setText("Grid resolution = "+str(self.canvas.gridResolution)+"x"+str(self.canvas.gridResolution))
        self.resolutionText.setFixedHeight(int(0.02*self.canvas.size))
        self.addWidget(self.resolutionText)

        self.drawGridButton = QtWidgets.QCheckBox('Draw Grid')
        self.drawGridButton.stateChanged.connect(self.toggleGrid)
        self.addWidget(self.drawGridButton)

        # Layout Chooser
        self.layoutChooser = QtWidgets.QComboBox()
        self.layoutChooser.addItems(['fluid_settings', 'fluid_constraints'])
        self.addWidget(self.layoutChooser)
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


        # Fluid Settings Layout
        self.frames = QtWidgets.QLineEdit(str(self.fluid.Nframes))
        self.frames.setValidator(QtGui.QIntValidator())
        self.frames.setMaxLength(4)
        self.frames.textChanged.connect(self.setFrames)
        self.fluidSimulationLayout.addRow("Frames", self.frames)

        self.dt = QtWidgets.QLineEdit(str(self.fluid.dt))
        dt_validator = QtGui.QDoubleValidator(0.001, 1.000, 3, notation=QtGui.QDoubleValidator.StandardNotation)
        dt_validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.dt.setValidator(dt_validator)
        self.dt.textChanged.connect(self.setTimestep)
        self.fluidSimulationLayout.addRow("Timestep", self.dt)

        self.boundary = QtWidgets.QComboBox()
        self.boundary.addItems([ 'dirichlet', 'neumann'])
        self.boundary.currentTextChanged.connect(self.setBoundary)
        self.fluidSimulationLayout.addRow("Boundary Conditions", self.boundary)

        self.source = QtWidgets.QCheckBox('Add Source')
        self.source.stateChanged.connect(self.setSource)
        self.fluidSimulationLayout.addWidget(self.source)
        self.sourceDuration = QtWidgets.QLineEdit(str(self.fluid.sourceDuration))
        self.sourceDuration.setValidator(QtGui.QIntValidator())
        self.sourceDuration.setMaxLength(4)
        self.sourceDuration.textChanged.connect(self.setSourceDuration)
        self.fluidSimulationLayout.addRow("Source Frames", self.sourceDuration)

        self.bakeFile = QtWidgets.QLineEdit(self.fluid.filename)
        self.bakeFile.textChanged.connect(self.setBakeFile)
        self.fluidSimulationLayout.addRow("File to bake", self.bakeFile)

        self.bakeButton = QtWidgets.QPushButton('Bake Simulation')
        self.bakeButton.clicked.connect(self.bake)
        self.fluidSimulationLayout.addWidget(self.bakeButton)

        self.playFile = QtWidgets.QLineEdit()
        self.playFile.textChanged.connect(self.setFileToPlay)
        self.fluidSimulationLayout.addRow("File to play", self.playFile)     

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
            self.fluid.d = tf.convert_to_tensor(self.canvas.initialDensity, dtype=tf.float32)
            self.fluid.layer.densities = [self.fluid.d.numpy()]

        # If we are not drawing constraint
        if index != 1:
            self.canvas.mode = ""
        else:
            self.combobox.setCurrentText("trajectory")
            self.canvas.mode = "trajectory"

    def setBakeFile(self, filename):
        self.fluid.filename = filename

    def setFileToPlay(self, text):
        if len(text.strip()) > 0:
            self.fluid.file_to_play = text
        else:
            self.fluid.file_to_play = None

    def setFrames(self, frames):
        self.fluid.Nframes = int(frames)

    def setTimestep(self, dt):
        self.fluid.dt = float(dt)

    def setBoundary(self, text):
        self.fluid.boundary = text    

    def setSource(self, state):
        if state == QtCore.Qt.Checked:
            self.fluid.useSource = True
        else:
            self.fluid.useSource = False
    
    def setSourceDuration(self, frame):
        self.fluid.sourceDuration = int(frame)+1

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
        self.canvas.clean()
        self.canvas.setGridResolution(int(np.sqrt(len(data["init_density"]))))
        self.fluid.setSize(int(np.sqrt(len(data["init_density"]))))
        self.canvas.hideGrid()
        if self.drawGridButton.isChecked():
            self.canvas.drawGrid()
        self.resolutionText.setText("Grid resolution = "+str(self.canvas.gridResolution)+"x"+str(self.canvas.gridResolution))

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

    def toggleGrid(self, state):
        if state == QtCore.Qt.Checked:
            self.canvas.drawGrid()
        else:
            self.canvas.hideGrid()

    def bake(self):
        print("Baking Simulation")
        lu, p, velocity_diff_LU, velocity_diff_P, scalar_diffuse_LU, scalar_diffuse_P = self.fluid.buildMatrices()
        file = "../bake/{filename}.json".format(filename=self.fluid.filename)
        self.fluid.bakeSimulation(lu, p, velocity_diff_LU, velocity_diff_P, scalar_diffuse_LU, scalar_diffuse_P, file)
        self.fluid.file_to_play = self.fluid.filename
        self.playFile.setText(self.fluid.file_to_play)

    def play(self):
        print("Play Simulation...")
        self.fluid.playDensity("../bake/")
