# This is part of the build process, it converts Qt Designer UI layouts to Python files
import os

def ConvertQt(fileName):
    print("Converting %s" % fileName)
    os.system("pyuic5 %s.ui -o %s.py" % (fileName, fileName))

print('Generating UI Python files')
ConvertQt("InputDialog")
ConvertQt("ProcessingDialog")
ConvertQt("OutputDialog")
