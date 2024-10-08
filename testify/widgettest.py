from PyQt5 import QtWidgets, QtCore

class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.scrollArea = QtWidgets.QScrollArea(self)
        self.scrollArea.setGeometry(10, 10, 400, 300)

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(0, 0, 800, 600)  # 设置内容区域的大小

        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        # 添加新的标签和复选框
        self.add_new_widgets(50, 50, 100, 100)

    def add_new_widgets(self, label_x, label_y, checkbox_x, checkbox_y):
        newlabel = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        newlabel.setEnabled(True)
        newlabel.setGeometry(QtCore.QRect(label_x, label_y, 200, 150))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(newlabel.sizePolicy().hasHeightForWidth())
        newlabel.setSizePolicy(sizePolicy)
        newlabel.setMinimumSize(QtCore.QSize(200, 150))
        newlabel.setMaximumSize(QtCore.QSize(200, 150))
        newlabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        newlabel.setStyleSheet("background-color: rgb(191, 191, 191);")
        newlabel.setAlignment(QtCore.Qt.AlignCenter)
        newlabel.setObjectName(f"newlabel_{len(self.scrollAreaWidgetContents.children())}")
        newlabel.setText("Label")

        newcheckBox = QtWidgets.QCheckBox(self.scrollAreaWidgetContents)
        newcheckBox.setGeometry(QtCore.QRect(checkbox_x, checkbox_y, 80, 20))
        newcheckBox.setObjectName(f"checkBox_{len(self.scrollAreaWidgetContents.children())}")

        print(f"newlabel_{len(self.scrollAreaWidgetContents.children())},x:{label_x},y:{label_y}")
        print(f"checkBox_{len(self.scrollAreaWidgetContents.children())},x:{checkbox_x},y:{checkbox_y}")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())