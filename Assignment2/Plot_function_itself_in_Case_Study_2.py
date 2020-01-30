"""
Visualize the function of case study 2.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import pyplot

X = np.linspace(0, 5, 100)     
Y = np.linspace(0, 5, 100)     
X, Y = np.meshgrid(X, Y) 

Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
fig = pyplot.figure() 
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)    

plt.show()



