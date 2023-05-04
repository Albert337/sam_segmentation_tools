from PyQt5.QtWidgets import QApplication, QLabel, QMenu, QAction, QMainWindow, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt,QEvent


class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        # 设置主窗口中心控件
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # 创建 label 和 list 控件
        self.label_2 = QLabel("点击我", self)
        self.label_2.setAlignment(Qt.AlignCenter)
        self.list_widget = QListWidget(self)

        # 添加控件到 main window 的中心
        layout.addWidget(self.label_2)
        layout.addWidget(self.list_widget)
        self.setCentralWidget(central_widget)

        # 用于保存鼠标点击的位置列表
        self.points = []

        # 设置 label 控件的事件过滤器
        self.label_2.installEventFilter(self)

        # 连接 list 控件的 itemClicked 信号槽
        self.list_widget.itemClicked.connect(self.itemClickedHandler)

        self.initUI()

    def initUI(self):

        self.setWindowTitle('Right-click Menu')
        self.setGeometry(300, 300, 350, 350)
        self.show()

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and \
            source == self.label_2:
            # 鼠标左键按下，并且点击的是 label_2
            pos = event.pos()  # 获取鼠标点击位置
            self.points.append(pos)  # 添加到列表中
            self.updateLabel()  # 更新 label 控件
            self.updateList()  # 更新列表控件
        return super().eventFilter(source, event)

    def updateLabel(self):
        # 获取最新的鼠标单击位置
        latest = self.points[-1]
        # 在 label 控件中显示该位置
        self.label_2.setText(f"最新单击位置：({latest.x()},{latest.y()})")

    def updateList(self):
        # 清空列表控件中的所有项
        self.list_widget.clear()
        # 添加最近添加的 5 个元素
        for point in reversed(self.points[-5:]):
            item = QListWidgetItem(f'({point.x()},{point.y()})')
            self.list_widget.addItem(item)

    def itemClickedHandler(self, item):
        # 在 label 控件中显示该 location
        text = item.text()
        self.label_2.setText(f"最新单击位置：{text}")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
