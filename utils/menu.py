import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import callbacksUI as callback
import canvas
import numpy as np

class Menu(QtWidgets.QVBoxLayout):

    def __init__(self):
        super().__init__()

        self.canvas = canvas.Canvas()

        self.resolutionText = QtWidgets.QLabel()
        self.resolutionText.setText("Grid resolution = "+str(self.canvas.gridResolution)+"x"+str(self.canvas.gridResolution))
        self.resolutionText.setFixedHeight(int(0.02*self.canvas.size))
        self.addWidget(self.resolutionText)

        
        self.combobox = QtWidgets.QComboBox()
        self.combobox.addItems(['trajectory', 'initial_density', 'target_density'])
        self.addWidget(self.combobox)
        # # Connect signals to the methods.
        # self.combobox.activated.connect(self.activated)
        self.combobox.currentTextChanged.connect(self.setMode)
        # self.combobox.currentIndexChanged.connect(self.index_changed)

        self.saveButton = QtWidgets.QPushButton('Save Config')
        self.saveButton.clicked.connect(self.save_config)
        self.addWidget(self.saveButton)

        self.loadButton = QtWidgets.QPushButton('Load Config')
        self.loadButton.clicked.connect(self.load_config)
        self.addWidget(self.loadButton)

        self.resetTrajButton = QtWidgets.QPushButton('Reset Trajectory')
        self.resetTrajButton.clicked.connect(self.reset_config)
        self.addWidget(self.resetTrajButton)

        self.drawGridButton = QtWidgets.QPushButton('Draw Grid')
        self.drawGridButton.clicked.connect(self.drawGrid)
        self.addWidget(self.drawGridButton)

        self.testButton = QtWidgets.QPushButton('Test Button')
        self.testButton.clicked.connect(self.test)
        self.addWidget(self.testButton)

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

    def drawGrid(self):
        self.canvas.drawGrid()

    def test(self):
        self.canvas.drawCell(5,0)
