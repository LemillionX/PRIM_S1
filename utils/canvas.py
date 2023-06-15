from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import numpy as np

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.size = 480

        # Mode attribute
        self.mode = "trajectory"

        # Grid attributes
        self.grid = QtWidgets.QLabel(self)
        grid_pixmap = QtGui.QPixmap(self.size,  self.size)
        grid_pixmap.fill(Qt.transparent)
        self.grid.setPixmap(grid_pixmap)
        self.gridResolution = 20
        self.blocSize = self.size/self.gridResolution

        # Initialise the pixel map
        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

        # Fluid attributes
        self.initialDensity = np.zeros(self.gridResolution*self.gridResolution)
        self.targetDensity = np.zeros(self.gridResolution*self.gridResolution)

        # Drawing attributes
        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')
        self.curves = []
        self.contour = []

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        if self.mode in ["trajectory", "initial_density"]:
            if self.last_x is None: # First event.
                self.last_x = e.x()
                self.last_y = e.y()
                if self.mode == "trajectory":
                    self.curves.append([[self.last_x, self.last_y]])
                if self.mode == "initial_density":
                    self.contour.append([[self.last_x, self.last_y]])
                return # Ignore the first time.


            if self.mode == "trajectory":
                painter = QtGui.QPainter(self.pixmap())
                p = painter.pen()
                p.setWidth(4)
                p.setColor(self.pen_color)
                painter.setPen(p)
                painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
                painter.end()
            if self.mode == "initial_density":
                i = e.x()//int(self.blocSize)
                j = self.gridResolution - 1 - e.y()//int(self.blocSize)
                self.drawCell(i,j,0, 255, 0)
                self.initialDensity[i+j*self.gridResolution] = 1
            self.update()

            # Update the origin for next time.
            self.last_x = e.x()
            self.last_y = e.y()
            if self.mode == "trajectory":
                self.curves[-1].append([self.last_x, self.last_y])
            if self.mode == "initial_density":
                self.contour[-1].append([[self.last_x, self.last_y]])

    def mouseReleaseEvent(self, e):
        if self.mode == "initial_density":
            print("Filling density")
            i = e.x()//int(self.blocSize)
            j = self.gridResolution - 1 - e.y()//int(self.blocSize)
            self.flood_fill(self.initialDensity, i+1, j)
            self.drawDensity(self.initialDensity, 0, 255, 0)
        self.last_x = None
        self.last_y = None

    def mousePressEvent(self, e):
        if self.mode == "initial_density":
            i = e.x()//int(self.blocSize)
            j = self.gridResolution - 1 - e.y()//int(self.blocSize)
            self.drawCell(i,j,255*self.initialDensity[i+j*self.gridResolution], 255, 255*self.initialDensity[i+j*self.gridResolution])
            self.initialDensity[i+j*self.gridResolution] = 1 - self.initialDensity[i+j*self.gridResolution]
        if self.mode == "target_density":
            i = e.x()//int(self.blocSize)
            j = self.gridResolution - 1 - e.y()//int(self.blocSize)
            self.drawCell(i,j,255, 255, 255*self.targetDensity[i+j*self.gridResolution])
            self.targetDensity[i+j*self.gridResolution] = 1 - self.targetDensity[i+j*self.gridResolution]


    def setGridResolution(self, resolution, drawGrid=False):
        self.gridResolution = resolution
        self.blocSize = self.size/self.gridResolution
        self.initialDensity = np.zeros(self.gridResolution*self.gridResolution)
        self.targetDensity = np.zeros(self.gridResolution*self.gridResolution)
        if drawGrid:
            self.hideGrid()
            self.drawGrid()
        self.update()

    def drawGrid(self):
        painter = QtGui.QPainter(self.grid.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(1)
        pen.setColor(Qt.gray)
        painter.setPen(pen)
        for i in range(self.gridResolution+1):
            painter.drawLine(int(self.blocSize*i), self.size, int(self.blocSize*i),0)
            painter.drawLine(0, int(self.blocSize*i), self.size, int(self.blocSize*i))
        painter.end()
        self.update()

    def hideGrid(self):
        grid_pixmap = QtGui.QPixmap(self.size,  self.size)
        grid_pixmap.fill(Qt.transparent)
        self.grid.setPixmap(grid_pixmap)        

    def drawCell(self, i, j, r=0,g=0,b=0, alpha=255):
        painter = QtGui.QPainter(self.pixmap())
        pen = QtGui.QPen()
        pen.setWidth(int(self.blocSize))
        pen.setColor(QtGui.QColor(int(r), int(g), int(b), int(alpha)))
        painter.setPen(pen)
        painter.drawPoint(int(self.blocSize*(i+0.5)), int(self.blocSize*(self.gridResolution-1 - j+0.5)))
        painter.end()
        self.update()

    def drawDensity(self, density, r,g,b):
        for i in range(self.gridResolution):
            for j in range(self.gridResolution):
                self.drawCell(i,j, r,g,b,int(255*density[i+j*self.gridResolution]))

    def setInitialDensity(self, density):
        self.initialDensity = np.array(density)
        self.drawDensity(self.initialDensity, 0,255,0)

    def setTargetDensity(self, density):
        self.targetDensity = np.array(density)
        self.drawDensity(self.targetDensity, 255,255,0)

    def drawCurve(self, curves):
        painter = QtGui.QPainter(self.pixmap())
        pen = painter.pen()
        pen.setWidth(4)
        painter.setPen(pen)
        for line in curves:
            for i in range(len(line)-1):
                painter.drawLine(line[i][0], line[i][1], line[i+1][0], line[i+1][1])
        painter.end()
        self.update()

    def setCurves(self, curves):
        self.curves = curves
        self.drawCurve(self.curves)

    def clean(self):
        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)
        self.curves = []

    def flood_fill(self, density, start_i, start_j):
        def fill(i,j,old,new):
            if density[i+j*self.gridResolution] != old:
                return
            elif density[i+j*self.gridResolution] == new:
                return
            else:
                density[i+j*self.gridResolution] = new
                neighbors = [(i-1, j-1), (i,j-1), (i+1,j-1),
                             (i-1, j),             (i+1, j),
                             (i-1, j+1), (i, j+1), (i+1,j+1)]
                for n in neighbors:
                    if  0 <= n[0] <= self.gridResolution -1 and 0 <= n[1] <= self.gridResolution -1:
                        fill(n[0], n[1], old, new)
        fill(start_i, start_j, 0, 1)
        return density
        
