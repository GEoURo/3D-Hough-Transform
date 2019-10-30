import numpy as np
from hough import hough
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

HT = hough(1, 1, 1)
HT.scan_points('./XYZcamera.mat')
X = HT.pointListX
Y = HT.pointListY
Z = HT.pointListZ
plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(X, Y, Z)
plt.show()