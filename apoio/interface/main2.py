import sys
import subprocess
from PyQt4 import QtGui, QtCore

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 300, 300)
        self.setWindowTitle("Reconhecimento de textos digitais e sua sintetizacao em linguagem natural humana")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))

        self.home()

    def home(self):
        self.btn = QtGui.QPushButton("Inicie", self)
        self.btn.clicked.connect(self.executar)
	self.btn.move(100,130)
	self.btn.setStyleSheet("background:#15CC40; border-radius:3px; border:0; color:white; padding:10px 30px; position:absolute;")

	self.but = QtGui.QPushButton("Feche", self)
        self.but.clicked.connect(self.fechar)
	self.but.move(100,180)
	self.but.setStyleSheet("background:#C20700; border-radius:3px; border:0; color:white; padding:100px 30px; position:absolute;")


	layout = QtGui.QVBoxLayout()
	layout.addStretch(1)
	layout.addWidget(self.btn)
	layout.addWidget(self.but)

	self.setLayout(layout)
        self.show()

    def executar(self):
	subprocess.call(["python", "geodet.py"])

    def fechar(self):
	sys.exit()
def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())


run()
