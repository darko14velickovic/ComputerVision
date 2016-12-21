from PySide import QtCore, QtGui


class CircleObject:

    def __init__(self, mode):
        self.center_radius = 5
        self.center_circle = None
        self.mode = mode
        self.circle = None
        self.is_start = False
        self.pen = QtGui.QPen("Black")
        self.alpha = 100

        if mode == "Good":
            self.active_color = QtGui.QColor(139, 195, 74, self.alpha)
            self.brush = QtGui.QBrush(self.active_color)
        if mode == "Bad":
            self.active_color = QtGui.QColor(244, 67, 54, self.alpha)
            self.brush = QtGui.QBrush(self.active_color)

        self.pen.setWidth(1)

    def left_click(self, position, scene):
        if not self.is_start:
            x = position.scenePos().x()
            y = position.scenePos().y()

            self.center_circle = QtGui.QGraphicsEllipseItem(x - self.center_radius/2, y - self.center_radius/2,
                                                            self.center_radius, self.center_radius)
            self.center_circle.setBrush(QtGui.QBrush(QtCore.Qt.black))
            self.center_circle.setPen(self.pen)

            self.circle = QtGui.QGraphicsEllipseItem(x, y, 1, 1)
            self.circle.setBrush(self.brush)
            self.circle.setPen(self.pen)

            scene.addItem(self.circle)
            scene.addItem(self.center_circle)
            scene.update()

            self.is_start = True
            return None
        else:
            self.is_start = False
            scene.removeItem(self.center_circle)
            return self.circle

    def right_click(self, position, scene, items):
        if self.is_start:
            scene.removeItem(self.circle)
            scene.removeItem(self.center_circle)
            self.is_start = False
        else:
            x = position.scenePos().x()
            y = position.scenePos().y()
            for i in range(0, len(items)):
                o = items[i]
                rect_o = o.rect()
                if rect_o.right() > x > rect_o.left():
                    if rect_o.top() < y < rect_o.bottom():
                        if self.active_color == items[i].brush().color():
                            items.remove(items[i])
                            scene.removeItem(o)
                            scene.update()
                            break

    def graphics_view_mouse_drag(self, drag, scene):
        if self.is_start:
            x = drag.scenePos().x()
            y = drag.scenePos().y()

            c = self.circle.rect().center()

            if x > c.x():
                new_x = x - c.x()
            else:
                new_x = c.x() - x

            if y > c.y():
                new_y = y - c.y()
            else:
                new_y = c.y() - y

            r = new_x
            if new_y > new_x:
                r = new_y

            x_dot = c.x() - r
            y_dot = c.y() - r

            self.circle.setRect(x_dot, y_dot, r*2, r*2)
