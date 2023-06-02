from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.size = 1000

        pixmap = QtGui.QPixmap(self.size,  self.size)
        pixmap.fill(Qt.white)
        self.setPixmap(pixmap)

        self.gridResolution = 20

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#000000')
        self.lineData = []

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            self.lineData.append([self.last_x, self.last_y])
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(4)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()
        self.lineData[-1].append([self.last_x, self.last_y])

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
        # print('\n'.join("Line #"+str(idx)+": "+str(element) for idx, element in enumerate(self.lineData)))

    def drawGrid(self):
        print("Drawing Grid")
        blocSize = self.size/self.gridResolution
        painter = QtGui.QPainter(self.pixmap())
        for i in range(0, self.gridResolution+1):
            painter.drawLine(int(blocSize*i), self.size, int(blocSize*i),0)
            painter.drawLine(0, int(blocSize*i), self.size, int(blocSize*i))
        painter.end()
        self.update()

    def drawCell(self, i, j, alpha=255):
        painter = QtGui.QPainter(self.pixmap())
        blocSize = self.size/self.gridResolution
        pen = QtGui.QPen()
        pen.setWidth(int(blocSize))
        pen.setColor(QtGui.QColor(255,0,0,alpha))
        painter.setPen(pen)
        painter.drawPoint(int(blocSize*(i+0.5)), int(blocSize*(j+0.5)))
        painter.end()
        self.update()
