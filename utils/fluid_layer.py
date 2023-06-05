from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import numpy as np

class FluidLayer(QtWidgets.QLabel):

    def __init__(self, size:int):
        super().__init__()
        self.size = size

        # Initialise the pixel map
        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

        # Drawing attributes
        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')
        self.curves = []


    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def drawCell(self, blocSize, gridResolution, i, j, r=0,g=0,b=0, alpha=255):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(int(blocSize))
        pen.setColor(QtGui.QColor(int(r), int(g), int(b), int(alpha)))
        painter.setPen(pen)
        painter.drawPoint(int(blocSize*(i+0.5)), int(blocSize*(gridResolution-1 - j+0.5)))
        painter.end()
        self.update()

    def clean(self):
        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)