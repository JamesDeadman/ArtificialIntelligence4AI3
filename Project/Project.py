import sys
import ntpath
import numpy as np
import math
import io
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage

from matplotlib import pyplot as plt
from PIL import Image 
from PIL import ImageQt 
from PIL import ImageOps
from PIL import ImageDraw
from random import seed
from random import random
from random import randint

#View layer
#Input dialog
from InputDialog import Ui_InputDialog

#Processing dialog
from ProcessingDialog import Ui_ProcessingDialog

#Output dialog
from OutputDialog import Ui_OutputDialog

EPSILON = .001

# A dialog section for a part item in the part item group area
class PartListItemView:
    def __init__(self, partItem, parentView, index):
        self.partItem = partItem
        self.parentView = parentView
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
        self.fileNameLabel.setText(self.partItem.Name)
        self.removeButton.setText("Remove")
        self.quantityLabel.setText("Qty:")

    def PlaceInLayout(self):
        self.parentView.GeometryListLayout.addWidget(self.PartFrame, self.partItem.Index, 0)

    def RemoveFromLayout(self):
        self.PartFrame.setParent(None)

#Presentation Layer
class PartListItemPresenter:
    @property
    def PartListItem(self):
        return self.partListItem

    @property
    def PartListItemView(self):
        return self.partListItemView

    def __init__(self, parentInputPresenter, partListItem, index):
        self.partListItem = partListItem
        self.parentInputPresenter = parentInputPresenter
        self.partListItemView = PartListItemView(partListItem, parentInputPresenter.InputView, index)
        self.partListItemView.CreateUIComponents()
        self.partListItemView.removeButton.clicked.connect(self.OnRemoveClicked)
        self.UpdateView()

    def OnRemoveClicked(self):
        self.partListItemView.RemoveFromLayout()
        self.parentInputPresenter.RemovePart(self)

    def UpdateModel(self):
        self.partListItem.Quantity = self.partListItemView.quantitySpinBox.value()

    def UpdateView(self):
        self.partListItemView.quantitySpinBox.setValue(self.partListItem.Quantity)

    def RemoveFromLayout(self):
        self.partListItemView.RemoveFromLayout()

    def PlaceInLayout(self):
        self.partListItemView.PlaceInLayout()


class InputPresenter:
    @property
    def InputView(self):
        return self.inputView

    def __init__(self, application, nestingRun, settings):
        self.application = application
        self.settings = settings
        self.nestingRun = nestingRun
        self.inputWindow = QtWidgets.QDialog()
        self.inputView = Ui_InputDialog()
        self.inputView.setupUi(self.inputWindow)
        self.inputView.CloseButton.clicked.connect(self.OnCloseClick)
        self.inputView.RunButton.clicked.connect(self.OnRunClick)
        self.inputView.BrowseButton.clicked.connect(self.OnBrowseClick)
        self.inputView.GeometryListLayout.setSpacing(0)

        for magnet in self.settings.Magnets:
            self.inputView.MagnetCombo.addItem(magnet.Name)

        for innerShapePolicy in self.settings.InnerShapePolicies:
            self.inputView.InnerShapeCombo.addItem(innerShapePolicy.Name)

        for openLoopPolicy in self.settings.OpenLoopPolicies:
            self.inputView.OpenLoopCombo.addItem(openLoopPolicy.Name)

        for flipPolicy in self.settings.FlipPolicies:
            self.inputView.AllowFlipCombo.addItem(flipPolicy.Name)

        self.partListItemPresenters = []
        self.nextItemIndex = 0
        
        self.UpdateView()

    def UpdateModel(self):
        self.settings.Population = self.inputView.PopulationSpinner.value()
        self.settings.Iterations = self.inputView.IterationSpinner.value()
        self.settings.MutationRate = self.inputView.MutationSpinner.value()
        self.settings.SpawnRate = self.inputView.SpawnSpinner.value()
        self.settings.Magnet = self.inputView.MagnetCombo.currentText()
        self.settings.OpenLoop = self.inputView.OpenLoopCombo.currentText()
        self.settings.InnerShapePolicy = self.inputView.InnerShapeCombo.currentText()
        self.settings.FlipPolicy = self.inputView.AllowFlipCombo.currentText()
        self.settings.MaxHeight = self.inputView.MaxHeightSpinner.value()
        self.settings.Kerf = self.inputView.KerfSpinner.value()
        self.settings.AngularStep = self.inputView.AngularStepSpinner.value()
        self.settings.MaxWidth = self.inputView.MaxWidthSpinner.value()
        self.settings.LinearStep = self.inputView.LinearStepSpinner.value()
        self.settings.Resolution = self.inputView.ResolutionSpinner.value()
        
        for partListItemPresenter in self.partListItemPresenters:
            partListItemPresenter.UpdateModel()

    def UpdateView(self):
        self.inputView.PopulationSpinner.setValue(self.settings.Population)
        self.inputView.IterationSpinner.setValue(self.settings.Iterations)
        self.inputView.MutationSpinner.setValue(self.settings.MutationRate)
        self.inputView.SpawnSpinner.setValue(self.settings.SpawnRate)
        self.inputView.MagnetCombo.setCurrentText(self.settings.Magnet)
        self.inputView.OpenLoopCombo.setCurrentText(self.settings.OpenLoopPolicy)
        self.inputView.InnerShapeCombo.setCurrentText(self.settings.InnerShapePolicy)
        self.inputView.AllowFlipCombo.setCurrentText(self.settings.FlipPolicy)
        self.inputView.MaxHeightSpinner.setValue(self.settings.MaxHeight)
        self.inputView.KerfSpinner.setValue(self.settings.Kerf)
        self.inputView.AngularStepSpinner.setValue(self.settings.AngularStep)
        self.inputView.MaxWidthSpinner.setValue(self.settings.MaxWidth)
        self.inputView.LinearStepSpinner.setValue(self.settings.LinearStep)
        self.inputView.ResolutionSpinner.setValue(self.settings.Resolution)

        for partListItemPresenter in self.partListItemPresenters:
            partListItemPresenter.UpdateView()

    def OnCloseClick(self):
        self.application.Terminate()

    def OnRunClick(self):
        self.UpdateModel()
        self.application.ExecuteNestingRun()

    def OnBrowseClick(self):
        fileNames, fileTypes = QFileDialog.getOpenFileNames(None, 'Open file',  'C:\\Temp',"Image files (*.bmp)")

        for fileName in fileNames:
            newPart = Part(self.nestingRun, fileName, 1)

        self.nestingRun.Batch.AddPart(newPart)

        partListItemPresenter = PartListItemPresenter(self, newPart, self.nextItemIndex)
        self.partListItemPresenters.append(partListItemPresenter)
        self.RefreshListItems()

    def RemovePart(self, partListItemPresenter):
        self.partListItemPresenters.remove(partListItemPresenter)
        self.RefreshListItems()

    def RefreshListItems(self):
        self.RemovePartListItemControlls()
        self.ReindexPartListItems()
        self.PlacePartListItemControlls()

    def ReindexPartListItems(self):
        i = 0
        for partListItemPresenter in self.partListItemPresenters:
            partListItemPresenter.PartListItem.Index = i
            i += 1
            
    def RemovePartListItemControlls(self):
        for partListItemPresenter in self.partListItemPresenters:
            partListItemPresenter.RemoveFromLayout()

    def PlacePartListItemControlls(self):
        for partListItemPresenter in self.partListItemPresenters:
            partListItemPresenter.PlaceInLayout()

    def Hide(self):
        self.inputWindow.hide()

    def Show(self):
        self.inputWindow.show()


class ProcessingPresenter:
    def __init__(self, application, nestingRun):
        self.application = application
        self.nestingRun = nestingRun
        self.processingWindow = QtWidgets.QDialog()
        self.processingView = Ui_ProcessingDialog()
        self.processingView.setupUi(self.processingWindow)
        self.processingView.CloseButton.clicked.connect(self.OnCloseClick)
        self.processingView.AbortButton.clicked.connect(self.OnAbortClick)

    def OnCloseClick(self):
        self.application.Terminate()

    def OnAbortClick(self):
        self.nestingRun.Abort()
        self.application.Navigator.ShowInput()

    def Hide(self):
        self.processingWindow.hide()

    def Show(self):
        self.processingWindow.show()


class OutputPresenter:
    def __init__(self, application):
        self.application = application
        self.outputWindow = QtWidgets.QDialog()
        self.outputView = Ui_OutputDialog()
        self.outputView.setupUi(self.outputWindow)
        self.outputView.CloseButton.clicked.connect(self.OnCloseClick)
        self.outputView.RestartButton.clicked.connect(self.OnRestartClick)
        self.outputView.SaveButton.clicked.connect(self.OnSaveClick)
        self.outputView.ResultGraphicsArea.paintEvent = self.PaintGraphicsArea
    def OnCloseClick(self):
        self.application.Terminate()
    
    def OnRestartClick(self):
        self.application.Navigator.ShowInput()

    def OnSaveClick(self):
        #TODO Show Save browse dialog
        raise NotImplementedError

    def PaintGraphicsArea(self, paintEvent):
        self.outputView.ResultGraphicsArea.setAutoFillBackground(True)
        painter = QtGui.QPainter(self.outputView.ResultGraphicsArea)

        displayImage = ImageOps.invert(self.application.nestingRun.DisplayImage)
        scaleFactor = min(self.outputView.ResultGraphicsArea.height() / displayImage.height, self.outputView.ResultGraphicsArea.width() / displayImage.width)
        displayImage = ImageOps.scale(displayImage, scaleFactor)
        painter.drawImage(0, 0, ImageQt.ImageQt(displayImage))

    def Hide(self):
        self.outputWindow.hide()

    def Show(self):
        self.outputWindow.show()
        

#Generic genetic algorithm model classes
class GeneticAlgorithmSettings:
    @property
    def Population(self):
        return self.population
    
    @Population.setter
    def Population(self, value):
        self.population = value
    
    @property
    def Iterations(self):
        return self.iterations
    
    @Iterations.setter
    def Iterations(self, value):
        self.iterations = value
    
    @property
    def MutationRate(self):
        return self.mutationRate
    
    @MutationRate.setter
    def MutationRate(self, value):
        self.mutationRate = value
    
    @property
    def SpawnRate(self):
        return self.spawnRate

    @SpawnRate.setter
    def SpawnRate(self, value):
        self.spawnRate = value

    def __init__(self):
        self.Population = 100
        self.Iterations = 100
        self.SpawnRate = 0.8
        self.MutationRate = 0.03
        

class Optimization:
    @property
    def Settings(self):
        return self.settings

    def __init__(self, settings):
        self.settings = settings


class Iteration:
    @property
    def Optimization(self):
        return self.optimization
    
    @property
    def Environment(self):
        return self.environtment

    def __init__(self, optimization, environment):
        self.optimization = optimization
        self.environtment = environment


class Environment:
    @property
    def Optimization(self):
        return self.optimization

    @property
    def Settings(self):
        return self.settings

    def __init__(self, optimization, settings):
        self.settings = settings
        self.optimization = optimization

    #Create a set of chromosomes for a given sequence
    def GenerateChromosomes(self, sequence):
        raise NotImplemented


class Population:
    @property
    def Environtment(self):
        return self.environment

    def __init__(self, environment):
        self.environment = environment


# A collection of related chromosomes
class Sequence:
    @property
    def Value(self):
        return self.value

    @property
    def Fitness(self):
        return self.fitness

    def __init__(self, population, parentA = None, parentB = None):
        self.population = population
        if(parentA is None): # this is a new sequence, generate chromosomes from the environment
            self.chromosomes = { x.GetId(): x for x in population.environment.GenerateChromosomes(self) }
        else: # this is a child of two parent sequences
            self.chromosomes = { }
            for k in parentA.chromosomes.keys() & parentB.chromosomes.keys():
                self.chromosomes[k] = Chromosome.Spawn(self, parentA.chromosomes[k], parentB.chromosomes[k])

    #Mutate each chromosome
    def Mutate(self):
        for chromosome in self.chromosomes.values():
             chromosome.Mutate()

    #Calculate the value of this sequence
    def CalculateValue(self):
        for chromosome in self.chromosomes.values():
             chromosome.CalculateValue()
        self.value = self.population.environment.Calculate(self.chromosomes)

    #Determine the fitness of this sequence
    def CalculateFitness(self):
        highest = self.population.GetHighestValue()
        lowest = self.population.GetLowestValue()
        self.fitness = (highest - self.Value) / (highest - lowest) + EPSILON


# Class to represent a single value in binary
class Chromosome:
    @property
    def Id(self):
        return self.Id

    @property
    def Value(self):
        return self.floatValue

    #Construct a random chromosome
    def __init__(self, sequence, id, dnaSize, minValue, maxValue, values = None):
        self.sequence = sequence
        self.id = id
        self.dnaSize = dnaSize
        self.minValue = minValue
        self.maxValue = maxValue

        if values is None:
            self.values = np.random.randint(2, size=(dnaSize))
        else:
            self.values = values   

    #Construct a random chromosome by crossing two parents
    @classmethod
    def Spawn(cls, sequence, parentA, parentB):
        values = parentA.values.copy()
        crossPoints = np.random.randint(0, 2, size=parentA.dnaSize).astype(np.bool)
        values[crossPoints] = parentB.values[crossPoints]
        return cls(sequence, parentA.id, parentA.dnaSize, parentA.minValue, parentA.maxValue, values)
    
    #Calculate the decimal value based on the binary value of the sequence
    def CalculateDecValue(self):
        self.decValue = self.values.dot(2 ** np.arange(self.dnaSize)[::-1])
        if(self.decValue < 0):
            print(self.decValue)

    #Calculate the float value based on the decimal value and the range
    def CalculateFloatValue(self):
        self.floatValue = self.decValue / float(2 ** self.dnaSize - 1) * (self.maxValue - self.minValue) + self.minValue

    #Calculate the decimal and float values
    def CalculateValue(self):
        self.CalculateDecValue()
        self.CalculateFloatValue()

    #Mutate each bit
    def Mutate(self):
        mutationRate = self.sequence.Population.Settings.MutationRate
        for point in range(self.dnaSize):
            if np.random.rand() < mutationRate:
                self.values[point] = 1 if self.values[point] == 0 else 0


#Nesting optimization model Classes
class Magnet:
    @property
    def Name(self):
        return self.name
        
    def __init__(self, name):
        self.name = name

        
class InnerShapePolicy:
    @property
    def Name(self):
        return self.name

    def __init__(self, name):
        self.name = name

        
class OpenLoopPolicy:
    @property
    def Name(self):
        return self.name
        
    def __init__(self, name):
        self.name = name


class FlipPolicy:
    @property
    def Name(self):
        return self.name

    def __init__(self, name):
        self.name = name

        
class NestingOptimizationSettings(GeneticAlgorithmSettings):
    Magnets = [
        Magnet("None"),
        Magnet("Top Left"),
        Magnet("Middle Left"),
        Magnet("Bottom Left"),
        Magnet("Top Center"),
        Magnet("Middle Center"),
        Magnet("Bottom Center"),
        Magnet("Top Right"),
        Magnet("Middle Right"),
        Magnet("Bottom Right")
    ]
    MagnetMap = { x.name: x for x in Magnets }

    InnerShapePolicies = [
        InnerShapePolicy("Allow"),
        InnerShapePolicy("Do not allow")
    ]
    InnerShapePolicyMap = { x.name: x for x in InnerShapePolicies }

    OpenLoopPolicies = [
        OpenLoopPolicy("Ignore"),
        OpenLoopPolicy("Close")
    ]
    OpenLoopPolicyMap = { x.name: x for x in OpenLoopPolicies }

    FlipPolicies = [
        FlipPolicy("Allow flip"),
        FlipPolicy("Do not flip")
    ]
    FlipPolicyMap = { x.Name: x for x in FlipPolicies }

    @property
    def Magnet(self):
        return self.magnet.Name
    
    @Magnet.setter
    def Magnet(self, value):
        self.magnet = NestingOptimizationSettings.MagnetMap[value]

    @property
    def OpenLoopPolicy(self):
        return self.openLoopPolicy.Name

    @OpenLoopPolicy.setter
    def OpenLoopPolicy(self, value):
        self.openLoopPolicy = NestingOptimizationSettings.OpenLoopPolicyMap[value]

    @property
    def InnerShapePolicy(self):
        return self.innerShape.Name

    @InnerShapePolicy.setter
    def InnerShapePolicy(self, value):
        self.innerShape = NestingOptimizationSettings.InnerShapePolicyMap[value]
        
    @property
    def FlipPolicy(self):
        return self.flipPolicy.Name

    @FlipPolicy.setter
    def FlipPolicy(self, value):     
        self.flipPolicy = NestingOptimizationSettings.FlipPolicyMap[value]   

    @property
    def MaxHeight(self):
        return self.maxHeight
        
    @MaxHeight.setter
    def MaxHeight(self, value):
        self.maxHeight = value

    @property
    def Kerf(self):
        return self.kerf
        
    @Kerf.setter
    def Kerf(self, value):
        self.kerf = value

    @property
    def AngularStep(self):
        return self.angularStep

    @AngularStep.setter
    def AngularStep(self, value):
        self.angularStep = value
        
    @property
    def MaxWidth(self):
        return self.maxWidth
        
    @MaxWidth.setter
    def MaxWidth(self, value):
        self.maxWidth = value

    @property
    def LinearStep(self):
        return self.linearStep

    @LinearStep.setter
    def LinearStep(self, value):
        self.linearStep = value
    
    @property
    def Resolution(self):
        return self.resolution
        
    @Resolution.setter
    def Resolution(self, value):
        self.resolution = value

    def __init__(self):
        super().__init__()
        self.Magnet = NestingOptimizationSettings.Magnets[0].Name
        self.OpenLoopPolicy = NestingOptimizationSettings.OpenLoopPolicies[0].Name
        self.InnerShapePolicy = NestingOptimizationSettings.InnerShapePolicies[0].Name
        self.FlipPolicy = NestingOptimizationSettings.FlipPolicies[0].Name
        self.MaxHeight = 0.0
        self.Kerf = 3.0
        self.AngularStep = 90.0 
        self.MaxWidth = 0.0
        self.LinearStep = 10.0
        self.Resolution = 2.5


class NestingRun(Optimization):
    @property
    def Batch(self):
        return self.batch

    @property
    def Material(self):
        return self.material

    @property
    def DisplayImage(self):
        return self.displayImage

    @property
    def Settings(self):
        return self.settings

    def __init__(self, settings):
        super().__init__(settings)
        self.batch = Batch(self, settings)
        
        self.batch.AddPart(Part(self.batch, "c:\\temp\\part1.bmp", 1))
        self.batch.AddPart(Part(self.batch, "c:\\temp\\part2.bmp", 2))
        self.batch.AddPart(Part(self.batch, "c:\\temp\\part3.bmp", 3))
        self.batch.AddPart(Part(self.batch, "c:\\temp\\part4.bmp", 4))
        self.batch.AddPart(Part(self.batch, "c:\\temp\\part5.bmp", 5))
        self.batch.AddPart(Part(self.batch, "c:\\temp\\part6.bmp", 6))

    
    def Run(self):
        self.continueRunning = True

        material = Material(self.batch)
        material.batch.CreateInstances()
        material.CreateVariants()

        #currentIteration = Iteration(self, self.batch)
        maxIteration = self.Settings.Iterations

        while (i < maxIteration and self.continueRunning):
            i += 1

            for variant in material.Variants
                variant.CreateRotatedImages()
                variant.CalculateSize()
                variant.InitBuffers()
                variant.CreatePartInstanceVariantImages()
                variant.FindBoundingBox()
                variant.CreateDisplayImage()

        #self.material = material
        #self.displayImage = materialInstance.DisplayImage
        #plt.imshow(materialInstance.DisplayImage)
        #plt.show()
       
        #i = 0


    def Abort(self):
        self.continueRunning = False


# Contains definiton and quantity of each part
class Batch(Environment):
    @property
    def Parts(self):
        return self.parts

    @property
    def Settings(self):
        return self.settings

    def __init__(self, nestingRun, settings):
        super().__init__(nestingRun, settings)
        self.parts = []
        self.partInstances = []

    def AddPart(self, part):
        self.parts.append(part)

    def RemovePart(self, part):
        self.parts.remove(part)

    def CreateInstances(self):
        for part in self.parts:
            for i in range(1, part.Quantity):
                self.partInstances.append(PartInstance(self.materialInstance, part))

    # Collect generate Part Instance Variants 
    def GenerateChromosomes(self, materialVariant):
        materialVariant.CreatePartInstanceVariants()

        for variant in materialVariant.PartInstanceVariants:
            for dimension in variant.Dimensions:
                materialVariant.chromosomes.addItem(dimension)


# Represents the entire material comprises the population
class MaterialVariant(Sequence):
    @property
    def PartInstanceVariants(self):
        return self.partInstanceVariants

    @property
    def MaterialImage(self) -> Image:
        return self.materialImage

    @property
    def OverlapImage(self) -> Image:
        return self.overlapImage

    @property
    def DisplayImage(self) -> Image:
        return self.displayImage

    @property
    def OriginX(self) -> int:
        return self.originX

    @property
    def OriginY(self) -> int:
        return self.originY

    @property
    def Width(self) -> int:
        return self.width

    @property
    def Height(self) -> int:
        return self.height

    def __init__(self, batch, population, parentA = None, parentB = None):
        super().__init__(population, parentA, parentB)
        self.population = population
        self.batch = batch
        self.partInstanceVariants = []

    def CreatePartInstanceVariants(self):
        for partInstance in self.batch.PartInstances:
            self.partInstanceVariants.append(PartInstanceVariant(self, partInstance))

    def InitBuffers(self):
        self.materialImage = Image.new('1', self.size)
        self.overlapImage = Image.new('1', self.size)
        self.displayImage = Image.new('RGB', self.size)

    def AddPartInstanceVariant(self, partInstance):
        self.partInstanceVariants.append(partInstance)

    def CalculateSize(self):
        minX = self.partInstanceVariants[0].MinX
        maxX = self.partInstanceVariants[0].MaxX
        minY = self.partInstanceVariants[0].MinY
        maxY = self.partInstanceVariants[0].MaxY

        for partInstance in self.partInstanceVariants[1::]:
            minX = min(minX, partInstance.MinX)
            maxX = max(maxX, partInstance.MaxX)
            minY = min(minY, partInstance.MinY)
            maxY = max(maxY, partInstance.MaxY)

        self.height = maxY - minY
        self.width = maxX - minX
        self.originX = minX
        self.originY = minY

        self.size = (self.width, self.height)

    def CreateRotatedImages(self):
        for instances in self.partInstanceVariants:
             instances.CreateRotatedImage()

    def CreatePartInstanceVariantImages(self):
        for variant in self.partInstanceVariants:
             variant.createImage()

    def CreateDisplayImage(self):
        self.displayImage.paste(self.materialImage)
        red = Image.new('RGB', self.size)
        redDraw = ImageDraw.Draw(red)
        redDraw.rectangle([(0, 0), red.size], fill = (255, 0, 0))
        self.displayImage.paste(self.materialImage)
        self.displayImage.paste(red, self.overlapImage)

    def FindBoundingBox(self):
        materialBuffer = np.array(self.materialImage)
        rows = np.any(materialBuffer, axis=1)
        cols = np.any(materialBuffer, axis=0)
        self.top, self.bottom = np.where(rows)[0][[0, -1]]
        self.right, self.left = np.where(cols)[0][[0, -1]]
        self.area = (self.bottom - self.top) * (self.left - self.right)



class Part:
    @property
    def File(self) -> str:
        return self.file
    
    @property
    def Quantity(self) -> int:
        return self.quantity

    @Quantity.setter
    def Quantity(self, value):
        self.quantity = value

    @property
    def Name(self):
        return self.name
    
    @property
    def Instances(self):
        return self.instances

    @property
    def Batch(self):
        return self.batch

    @property
    def Image(self) -> Image:
        return self.image

    def __init__(self, batch, fileName, quantity):
        self.batch = batch
        self.file = fileName
        self.quantity = quantity
        self.name = ntpath.basename(fileName)
        self.instances = []
        self.quantity = quantity
        self.image = Image.open(self.file)

    def InitInstances(self):
        for i in range(1, self.quantity):
            self.instances.append(PartInstance(self, i))


# An instance of a single part (one for each quantity of part)
class PartInstance:
    @property
    def Index(self) -> int:
        return self.index

    def __init__(self, part, index):
        self.part = part
        self.index = index


# A variant of a part instance, one for each population member of part instance
class PartInstanceVariant():
    @property
    def X(self) -> int:
        return self.x
    
    @property
    def Y(self) -> int:
        return self.y

    @property
    def R(self) -> int:
        return self.r

    @property
    def F(self) -> bool:
        return self.f

    @property
    def MinX(self) -> int:
        return int(self.X - self.Width / 2)

    @property
    def MaxX(self) -> int:
        return int(self.X + self.Width / 2)

    @property
    def MinY(self) -> int:
        return int(self.Y - self.Height / 2) 

    @property
    def MaxY(self) -> int:
        return int(self.Y + self.Height / 2)

    @property
    def Width(self) -> int:
        return int(self.rotatedImage.width)

    @property
    def Height(self) -> int:
        return int(self.rotatedImage.height)

    @property
    def RotatedImage(self) -> Image:
        return self.rotatedImage

    def __init__(self, batch, materialVariant, partInstance, population):
        super().__init__(population)
        self.settings:NestingOptimizationSettings = settings
        self.batch = batch
        self.materialVariant = materialVariant
        self.partInstance = partInstance

        idPrefix = partInstance.Part.Name + "~" + partInstance.Index
        xSize = math.ceil(math.log2(self.settings.MaxWidth / self.settings.LinearStep))
        ySize = math.ceil(math.log2(self.settings.MaxHeight / self.settings.LinearStep))
        rSize = math.ceil(math.log2(360 / self.settings.AngularStep))
    
        self.x = Dimension(self, materialVariant, idPrefix + "x", xSize, 0, self.settings.MaxWidth)
        self.y = Dimension(self, materialVariant, idPrefix + "y", ySize, 0, self.settings.MaxHeight)
        self.r = Dimension(self, materialVariant, idPrefix + "r", rSize, 0, self.settings.MaxWidth)
        
        if self.settings.FlipPolicy.Name == FlipPolicies_AllowFlip.Name:
            self.f = Dimension(self, materialVariant, idPrefix + "f", 1, 0, 1)
        else:
            self.f = Dimension(self, materialVariant, idPrefix + "f", 0, 0, 0)

        self.chromosomes.addItem(self.x)
        self.chromosomes.addItem(self.y)
        self.chromosomes.addItem(self.r)
        self.chromosomes.addItem(self.f)

    def CreateRotatedImage(self):
        image = self.part.Image

        if(self.f):
            image = ImageOps.flip(image)
        self.rotatedImage = image.rotate(self.r, Image.NEAREST, expand = 1)
        self.size = self.rotatedImage.size

    def createImage(self):
        # Overlap
        location = (int(self.x - self.Width / 2 - self.materialInstance.OriginX), int(self.y - self.Height / 2 - self.materialInstance.OriginY))
        self.overlapMask = Image.new('1', self.materialInstance.OverlapImage.size)
        self.overlapMask.paste(self.rotatedImage, location, self.rotatedImage)
        self.materialInstance.OverlapImage.paste(self.materialInstance.MaterialImage, self.overlapMask)

        # Place in material image
        self.materialInstance.MaterialImage.paste(self.rotatedImage, location, self.rotatedImage)


class Dimension(Chromosome):
    def __init__(self, materialVariant, id, dnaSize, minValue, maxValue):
        super().__init__(materialVariant, id, dnaSize, minValue, maxValue)


class RasterImage:
    def __init__(self):
        super().__init__()

        
class VectorImage:
    def __init__(self):
        super().__init__()


class Material:
    @property
    def Variants(self):
        return self.variants

    def __init__(self, batch):
        self.batch = batch
        self.population = Population(self.batch)
        self.variants = []

    def CreateVariants(self):
        for i in range(1, self.batch.Settings.Population):
            self.variant = MaterialVariant(self.batch, self.population)


#Utility and service classes
class ImageLoader:
    def LoadImage(self, fileName):
        raise NotImplementedError
        #TODO: Load a vector image

        
class ImageSaver:
    def SaveImage(self, vectorImage, fileName):
        raise NotImplementedError

        
class ImageConverter:
    def ConvertVectorToRaster(self):
        raise NotImplementedError

class ImageBliter:
    def BlitImage(self, sourceImage, destinationImage, x, y, r, f):
        raise NotImplementedError


#Top level application class
class Navigator:
    def __init__(self, inputPresenter, processingPresenter, outputPresenter):
        self.inputPresenter = inputPresenter
        self.processingPresenter = processingPresenter
        self.outputPresenter = outputPresenter

    def ShowInput(self):
        self.processingPresenter.Hide()
        self.outputPresenter.Hide()
        self.inputPresenter.Show()

    def ShowProcessing(self):
        self.inputPresenter.Hide()
        self.outputPresenter.Hide()
        self.processingPresenter.Show()

    def ShowOutput(self):
        self.inputPresenter.Hide()
        self.processingPresenter.Hide()
        self.outputPresenter.Show()


#Enumeration classes
class Application:
    @property
    def Settings(self):
        return self.settings
        
    @property
    def Navigator(self):
        return self.navigator

    def __init__(self, arguments):
        self.settings = NestingOptimizationSettings()
        self.nestingRun = NestingRun(self.settings)
        self.qtApplication = QtWidgets.QApplication(arguments)
        self.inputPresenter = InputPresenter(self, self.nestingRun, self.settings)
        self.processingPresenter = ProcessingPresenter(self, self.nestingRun)
        self.outputPresenter = OutputPresenter(self)
        self.navigator = Navigator(self.inputPresenter, self.processingPresenter, self.outputPresenter)

    def ExecuteNestingRun(self):
        self.Navigator.ShowProcessing()
        self.nestingRun.Run()
        self.Navigator.ShowOutput()

    def Run(self): 
        self.Navigator.ShowInput()
        sys.exit(self.qtApplication.exec_())

    def Terminate(self):
        sys.exit()

if __name__ == "__main__":
    application = Application(sys.argv)
    application.Run()
