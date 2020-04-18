# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'OutputDialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_OutputDialog(object):
    def setupUi(self, OutputDialog):
        OutputDialog.setObjectName("OutputDialog")
        OutputDialog.resize(681, 491)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(OutputDialog.sizePolicy().hasHeightForWidth())
        OutputDialog.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        OutputDialog.setFont(font)
        self.SaveButton = QtWidgets.QPushButton(OutputDialog)
        self.SaveButton.setGeometry(QtCore.QRect(10, 440, 121, 41))
        self.SaveButton.setObjectName("SaveButton")
        self.ResultGroup = QtWidgets.QGroupBox(OutputDialog)
        self.ResultGroup.setGeometry(QtCore.QRect(10, 10, 661, 421))
        self.ResultGroup.setObjectName("ResultGroup")
        self.ResultGraphicsArea = QtWidgets.QWidget(self.ResultGroup)
        self.ResultGraphicsArea.setGeometry(QtCore.QRect(10, 20, 641, 391))
        self.ResultGraphicsArea.setObjectName("ResultGraphicsArea")
        self.CloseButton = QtWidgets.QPushButton(OutputDialog)
        self.CloseButton.setGeometry(QtCore.QRect(550, 440, 121, 41))
        self.CloseButton.setObjectName("CloseButton")
        self.RestartButton = QtWidgets.QPushButton(OutputDialog)
        self.RestartButton.setGeometry(QtCore.QRect(420, 440, 121, 41))
        self.RestartButton.setObjectName("RestartButton")

        self.retranslateUi(OutputDialog)
        QtCore.QMetaObject.connectSlotsByName(OutputDialog)

    def retranslateUi(self, OutputDialog):
        _translate = QtCore.QCoreApplication.translate
        OutputDialog.setWindowTitle(_translate("OutputDialog", "Nesting"))
        self.SaveButton.setText(_translate("OutputDialog", "&Save"))
        self.ResultGroup.setTitle(_translate("OutputDialog", "Result"))
        self.CloseButton.setText(_translate("OutputDialog", "&Close"))
        self.RestartButton.setText(_translate("OutputDialog", "&Restart"))
