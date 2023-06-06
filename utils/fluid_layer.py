from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QTimer
import numpy as np

class FluidLayer(QtWidgets.QLabel):

    def __init__(self, size:int, grid_size:int):
        super().__init__()
        self.size = size
        self.grid_size = grid_size

        # Initialise the pixel map
        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

        # Drawing attributes
        self.last_x, self.last_y = None, None
        self.densities = None
        self.current_frame = 0
        self.timer = QTimer(self)
        self.fps = 30
        self.drawGrid = False
        self.r, self.g, self.b = (255, 0, 0)

    def setRGB(self, color:QtGui.QColor):
        self.r, self.g, self.b, _ = color.getRgb()
        self.update()

    def clean(self):
        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

    def paintEvent(self, e: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.fillRect(e.rect(), Qt.white)
        
        blocSize = int(self.size/self.grid_size)

        if self.densities is not None:
            density = self.densities[self.current_frame]
            pen = QtGui.QPen()
            pen.setWidth(blocSize)
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    pen.setColor(QtGui.QColor(self.r, self.g, self.b, int(255*density[i+j*self.grid_size])))
                    painter.setPen(pen)
                    painter.drawPoint(int(blocSize*(i+0.5)), int(blocSize*(self.grid_size-1 - j+0.5)))

        if self.drawGrid:
            pen = QtGui.QPen()
            pen.setWidth(1)
            pen.setColor(Qt.gray)
            painter.setPen(pen)
            for i in range(self.grid_size+1):
                painter.drawLine(blocSize*i, 0, blocSize*i, self.size)
                painter.drawLine(0, blocSize*i, self.size, blocSize*i)
        painter.end()