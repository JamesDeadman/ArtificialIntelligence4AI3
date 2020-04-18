import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image 
from PIL import ImageOps
from PIL import ImageDraw
from random import seed
from random import random
from random import randint

class Part:
    @property
    def Quantity(self):
        return self.quantity

    @property
    def Image(self) -> Image:
        return self.image

    def __init__(self, fileName, quantity):
        self.fileName = fileName
        self.quantity = quantity
        self.image = Image.open(fileName)


class Mataterial:
    def __init__(self):
        self.parts = []

    def AddPart(self, part):
        self.parts.append(part)

    def CreateInstances(self):
        self.materialInstance = MaterialInstance(self)
        seed()
        for part in self.parts:
            for i in range(1, part.Quantity):
                x = randint(0, 500)
                y = randint(0, 500)
                r = randint(0, 360)
                f = random() > 0.5
                self.materialInstance.AddPartInstance(PartInstance(self.materialInstance, part, x, y, r, f))
        return self.materialInstance


class MaterialInstance:
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

    def __init__(self, material):
        self.material = material
        self.partInstances = []

    def InitBuffers(self):
        self.materialImage = Image.new('1', self.size)
        self.overlapImage = Image.new('1', self.size)
        self.displayImage = Image.new('RGB', self.size)

    def AddPartInstance(self, partInstance):
        self.partInstances.append(partInstance)

    def CalculateSize(self):
        minX = self.partInstances[0].MinX
        maxX = self.partInstances[0].MaxX
        minY = self.partInstances[0].MinY
        maxY = self.partInstances[0].MaxY

        for partInstance in self.partInstances[1::]:
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
        for instances in self.partInstances:
             instances.CreateRotatedImage()

    def CreatePartInstanceImages(self):
        for instances in self.partInstances:
             instances.createImage()

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


class PartInstance:
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

    def __init__(self, materialInstance:MaterialInstance, part:Part, x: int, y: int, r: float, f: bool):
        self.materialInstance = materialInstance
        self.part = part
        self.x = x
        self.y = y
        self.r = r
        self.f = f

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


def main():
    material = Mataterial()

    material.AddPart(Part("c:\\temp\\part1.bmp", 1))
    material.AddPart(Part("c:\\temp\\part2.bmp", 2))
    material.AddPart(Part("c:\\temp\\part3.bmp", 3))
    material.AddPart(Part("c:\\temp\\part4.bmp", 4))
    material.AddPart(Part("c:\\temp\\part5.bmp", 5))
    material.AddPart(Part("c:\\temp\\part6.bmp", 6))
    
    materialInstance = material.CreateInstances()
    materialInstance.CreateRotatedImages()
    materialInstance.CalculateSize()
    materialInstance.InitBuffers()
    materialInstance.CreatePartInstanceImages()
    materialInstance.FindBoundingBox()
    materialInstance.CreateDisplayImage()

    plt.imshow(materialInstance.DisplayImage)
    plt.show()


if __name__ == "__main__": 
    main() 