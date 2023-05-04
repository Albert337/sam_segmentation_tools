"""
Override(覆盖) 槽函数
"""
"""
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

class OverrideSlot(QWidget):
    def __init__(self):
        super(OverrideSlot, self).__init__()
        self.setWindowTitle("Override(覆盖) 槽函数")

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_Alt:
            self.setWindowTitle("按下Alt键")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = OverrideSlot()
    main.show()

    sys.exit(app.exec_())
"""

from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QAction, QMainWindow, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt,QEvent

class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        # 创建 label 和 list 控件
        self.label_2 = QLabel("点击我", self)
        self.label_2.setGeometry(50, 50, 200, 50)
        self.label_2.setMouseTracking(True)
        self.label_2.installEventFilter(self)
        self.list_widget = QListWidget(self)
        self.list_widget.setGeometry(50, 110, 200, 200)

        self.points = []  # 用于保存鼠标点击的位置列表

        self.initUI()

    def initUI(self):

        self.setWindowTitle('Right-click Menu')
        self.setGeometry(300, 300, 350, 350)
        self.show()

    def eventFilter(self, source, event):

        #if (event.type() == Qt.MouseButtonPress and event.button() == Qt.LeftButton and source == self.label_2):
        if (event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and source==self.label_2):
            # 鼠标左键按下，并且点击的是 label_2
            
            pos = event.pos()  # 获取鼠标点击位置
            self.points.append(pos)  # 添加到列表中
            self.updateListWidget()  # 更新列表控件

        return super().eventFilter(source, event)

    def mousePressEvent(self, event):

        if event.button() == Qt.RightButton:
            # 鼠标右键按下，弹出菜单
            menu = QMenu(self)

            # 添加菜单项
            for point in reversed(self.points):
                action = QAction(f'({point.x()},{point.y()})', self)
                menu.addAction(action)

            # 连接菜单项的触发信号和槽函数
            menu.triggered.connect(self.menuActionTriggered)

            # 显示菜单
            menu.exec_(event.globalPos())

    def updateListWidget(self):
        # 清空列表控件中的所有项
        self.list_widget.clear()
        # 添加最近添加的 5 个元素
        for point in reversed(self.points[-5:]):
            item = QListWidgetItem(f'({point.x()},{point.y()})')
            self.list_widget.addItem(item)

    def menuActionTriggered(self, action):
        # 菜单项触发时，获取其文本，更新列表控件
        text = action.text()
        self.label_2.setText(f'最后点击位置：{text}')
        self.updateListWidget()

if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
