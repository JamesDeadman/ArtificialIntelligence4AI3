import sys
from array import *
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets

from InputDialog import Ui_InputDialog
from ProcessingDialog import Ui_ProcessingDialog
from OutputDialog import Ui_OutputDialog

nextItemIndex = 1
partItemViews = []

# A dialog section for a part item in the part item group area
class PartItemView:
    def __init__(self, partItem, parentLayout, index):
        self.partItem = partItem
        self.parentLayout = parentLayout
        self.index = index

    def CreateUIComponents(self):
        self.PartFrame = QtWidgets.QFrame()
        self.PartFrame.setGeometry(QtCore.QRect(0, self.index * 17, 311, 21))
        self.PartFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.PartFrame.setFrameShadow(QtWidgets.QFrame.Raised)

        self.fileNameLabel = QtWidgets.QLabel(self.PartFrame)
        self.fileNameLabel.setGeometry(QtCore.QRect(0, 0, 151, 16))
        self.fileNameLabel.setObjectName("fileNameLabel%s" % self.index)

        self.quantitySpinBox = QtWidgets.QSpinBox(self.PartFrame)
        self.quantitySpinBox.setGeometry(QtCore.QRect(200, 0, 51, 22))
        self.quantitySpinBox.setObjectName("qtySpinBox%s" % self.index)

        self.removeButton = QtWidgets.QPushButton(self.PartFrame)
        self.removeButton.setGeometry(QtCore.QRect(250, 0, 61, 21))
        self.removeButton.setObjectName("removeButton%s" % self.index)

        self.quantityLabel = QtWidgets.QLabel(self.PartFrame)
        self.quantityLabel.setGeometry(QtCore.QRect(160, 0, 31, 16))
        self.quantityLabel.setObjectName("quantityLabel")

        self.quantitySpinBox.setValue(self.partItem.quantity)
        self.fileNameLabel.setText(self.partItem.GetName())
        self.removeButton.setText("Remove")
        self.quantityLabel.setText("Qty:")
        
        self.parentLayout.addWidget(self.PartFrame)

class PartItem:
    def __init__(self, name):
        self.name = name
        self.quantity = 1

    def GetName(self):
        return self.name





def OnCloseClick():
    sys.exit()

def OnRunClick():
    inputDialogWindow.hide()
    processingDialogWindow.show()

    print ("Value: %s" % inputDialogView.IterationSpinner.value())

def OnBrowseClick():
    global nextItemIndex
    fname, ftype = QFileDialog.getOpenFileName(None, 'Open file',  'C:\\Users\\Public\\Pictures\\Sample Pictures',"Image files (*.jpg *.gif)")
    partItem = PartItem(fname)

    nextItemIndex += 1
    partItemView = PartItemView(partItem, inputDialogView.GeometryListLayout, nextItemIndex)
    partItemView.CreateUIComponents()

    partItemViews.append(partItemView)
    print (fname)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    inputDialogWindow = QtWidgets.QDialog()
    inputDialogView = Ui_InputDialog()
    inputDialogView.setupUi(inputDialogWindow)
    inputDialogView.CloseButton.clicked.connect(OnCloseClick)
    inputDialogView.RunButton.clicked.connect(OnRunClick)
    inputDialogView.BrowseButton.clicked.connect(OnBrowseClick)

    inputDialogView.MagnetCombo.addItem("Top Left")
    inputDialogView.OpenLoopCombo.addItem("Ignore")
    inputDialogView.InnerShapeCombo.addItem("Yes")

    processingDialogWindow = QtWidgets.QDialog()
    processingDialogView = Ui_ProcessingDialog()
    processingDialogView.setupUi(processingDialogWindow)
    processingDialogView.CloseButton.clicked.connect(OnCloseClick)
    #processingDialogView.RunButton.clicked.connect(OnRunClick)
    #processingDialogView.BrowseButton.clicked.connect(OnBrowseClick)

    outputDialogWindow = QtWidgets.QDialog()
    outputDialogView = Ui_OutputDialog()
    outputDialogView.setupUi(outputDialogWindow)
    outputDialogView.CloseButton.clicked.connect(OnCloseClick)
    #outputDialogView.RunButton.clicked.connect(OnRunClick)
    #outputDialogView.BrowseButton.clicked.connect(OnBrowseClick)

    inputDialogWindow.show()
    
    sys.exit(app.exec_())
