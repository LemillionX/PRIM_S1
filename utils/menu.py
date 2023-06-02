import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import callbacksUI as callback
import canvas

class Menu(QtWidgets.QVBoxLayout):

    def __init__(self):
        super().__init__()

        self.canvas = canvas.Canvas()

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

    def save_config(self):
        print("Saving file")

    def load_config(self):
        print("Load file")
        data = callback.loadFromJSON()
        self.canvas.clean()
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