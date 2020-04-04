# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ProcessingDialog.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ProcessingDialog(object):
    def setupUi(self, ProcessingDialog):
        ProcessingDialog.setObjectName("ProcessingDialog")
        ProcessingDialog.resize(621, 171)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ProcessingDialog.sizePolicy().hasHeightForWidth())
        ProcessingDialog.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        ProcessingDialog.setFont(font)
        self.GoutputGraphicsGroup = QtWidgets.QGroupBox(ProcessingDialog)
        self.GoutputGraphicsGroup.setGeometry(QtCore.QRect(10, 10, 601, 101))
        self.GoutputGraphicsGroup.setObjectName("GoutputGraphicsGroup")
        self.GeneticAlgorithmProgress = QtWidgets.QProgressBar(self.GoutputGraphicsGroup)
        self.GeneticAlgorithmProgress.setGeometry(QtCore.QRect(10, 60, 581, 23))
        self.GeneticAlgorithmProgress.setProperty("value", 24)
        self.GeneticAlgorithmProgress.setObjectName("GeneticAlgorithmProgress")
        self.MessageLabel = QtWidgets.QLabel(self.GoutputGraphicsGroup)
        self.MessageLabel.setGeometry(QtCore.QRect(10, 30, 561, 16))
        self.MessageLabel.setText("")
        self.MessageLabel.setObjectName("MessageLabel")
        self.CloseButton = QtWidgets.QPushButton(ProcessingDialog)
        self.CloseButton.setGeometry(QtCore.QRect(490, 120, 121, 41))
        self.CloseButton.setObjectName("CloseButton")
        self.AbortButton = QtWidgets.QPushButton(ProcessingDialog)
        self.AbortButton.setGeometry(QtCore.QRect(360, 120, 121, 41))
        self.AbortButton.setObjectName("AbortButton")

        self.retranslateUi(ProcessingDialog)
        QtCore.QMetaObject.connectSlotsByName(ProcessingDialog)

    def retranslateUi(self, ProcessingDialog):
        _translate = QtCore.QCoreApplication.translate
        ProcessingDialog.setWindowTitle(_translate("ProcessingDialog", "Nesting"))
        self.GoutputGraphicsGroup.setTitle(_translate("ProcessingDialog", "Genetic Algorithm"))
        self.CloseButton.setText(_translate("ProcessingDialog", "&Close"))
        self.AbortButton.setText(_translate("ProcessingDialog", "&Abort"))
