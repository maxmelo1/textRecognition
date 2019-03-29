import sys
from PyQt4 import QtGui
from PyQt4.uic import loadUiType
 
Ui_MainWindow, QMainWindow = loadUiType('interface.ui')
 
class Main(QMainWindow, Ui_MainWindow) :
	def __init__(self, ) :
		super(Main, self).__init__()
		self.setupUi(self)

	def home(self):
		self.ini = QtGui.QPushButton(self)
        	self.ini.clicked.connect(self.executar)

	def executar(self):
		print("adadada")
		subprocess.call(["python", "geodet.py"])

if __name__ == '__main__' :
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
