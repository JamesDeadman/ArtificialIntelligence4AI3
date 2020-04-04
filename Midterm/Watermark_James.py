import cv2
import numpy as np
from matplotlib import pyplot as plt

FILE_PATH = r'c:\\temp\\WatermarkCat.jpg'
THRESHOLD_VALUE = 181  # Darkest value found in the watermark
BACKGROUND_VALUE = 255

# Load the original image
watermarkCat = plt.imread(FILE_PATH)

# Filter the image using a threshold filter
ret, cleanCat = cv2.threshold(watermarkCat, THRESHOLD_VALUE, BACKGROUND_VALUE, cv2.THRESH_BINARY)

# Display the image
plt.imshow(cleanCat)
plt.show()
