"""
Visualize a function.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import pyplot


X = np.linspace(-1, 1, 100)     
Y = np.linspace(-1, 1, 100)     
X, Y = np.meshgrid(X, Y) 

Z = (3 * X**2 + np.sin(5 * np.pi * X)) + (3 * Y**4 + np.cos(3 * np.pi * Y)) + 10
fig = pyplot.figure() 
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
  cmap=cm.nipy_spectral, linewidth=0.08,
  antialiased=True)    

plt.show()
