from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import callbacksUI as callback
import canvas
import numpy as np

class Menu(QtWidgets.QVBoxLayout):

    def __init__(self):
        super().__init__()

        self.canvas = canvas.Canvas()

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
        self.fluidConstraintsLayout = QtWidgets.QVBoxLayout(self.fluid_constraint_container)
        self.fluidSimulationLayout = QtWidgets.QVBoxLayout(self.fluid_simulation_container)

        # Fluid Constraints Layout
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(['trajectory', 'initial_density', 'target_density'])
        self.fluidConstraintsLayout.addWidget(self.combobox)
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
        self.testButton = QtWidgets.QPushButton('Test Button')
        self.testButton.clicked.connect(self.test)
        self.fluidSimulationLayout.addWidget(self.testButton)

        # Stack Layouts
        self.stacked_layout = QtWidgets.QStackedLayout()
        self.stacked_layout.addWidget(self.fluid_simulation_container)
        self.stacked_layout.addWidget(self.fluid_constraint_container)
        self.addLayout(self.stacked_layout)

    def setLayout(self, index):
        print("Changing to Layout #"+str(index))
        self.stacked_layout.setCurrentIndex(index)


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
        self.canvas.hideGrid()
        if self.drawGridButton.isChecked():
            self.canvas.drawGrid()
        self.resolutionText.setText("Grid resolution = "+str(self.canvas.gridResolution)+"x"+str(self.canvas.gridResolution))

        self.canvas.setInitialDensity(data["init_density"])
        self.canvas.setTargetDensity(data["target_density"])
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

    def test(self):
        print("Hello World !")
