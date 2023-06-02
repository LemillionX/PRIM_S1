import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import canvas
from menu import *

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.fluidCanvas = canvas.Canvas()

        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout()
        w.setLayout(l)
        l.addWidget(self.fluidCanvas)

        palette = QtWidgets.QHBoxLayout()
        l.addLayout(palette)

        menu = Menu()
        menu.setCanvas(self.fluidCanvas)
        l.addLayout(menu)


        self.setCentralWidget(w)



app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()