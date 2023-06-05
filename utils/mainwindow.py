import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import canvas
from menu import *

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        w = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout()
        w.setLayout(l)

        self.menu = Menu(self)

        self.stacked_layout = QtWidgets.QStackedLayout()
        self.stacked_layout.addWidget(self.menu.fluid.layer)
        self.stacked_layout.addWidget(self.menu.canvas)
        l.addLayout(self.stacked_layout)


        l.addLayout(self.menu)


        self.setCentralWidget(w)
        self.setWindowTitle("PRIM - User-guided smoke simulation")



app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()