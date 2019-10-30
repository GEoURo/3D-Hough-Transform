import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin
from mpl_toolkits.mplot3d import Axes3D


class Plane_generator(object):
    def __init__(self):
        self.rho_list = None
        self.phi_list = None
        self.theta_list = None
        return
        

    def create_planes(self, plane_num=4, rho_range=5):
        """
        Generate planes
        Params:
        plane_num: Number of plane
        rho_range: Range of rho value
        """
        self.rho_list = np.random.choice(np.arange(-rho_range, rho_range, 2), size=plane_num)
        self.phi_list = np.random.choice(np.arange(-np.pi/2, np.pi/2, 0.5), size=plane_num)
        self.theta_list = np.random.choice(np.arange(0, np.pi, 0.5), size=plane_num)
        return

    def get_planes(self):
        return self.theta_list, self.phi_list, self.rho_list

    def to_pointclouds(self, pnum=100, xrange=50, yrange=50, biasrange=0.):
        """
        Generate point clouds based on the planes
        Params:
        pnum: points for each plane
        x/yrange: Range of X, Y for the point cloud
        biasrange: Bias to be added to each point, set to 0 by default

        Return:
        A numpy array of shape (N, 3)
        N is the total number of points
        """
        point_clouds = np.zeros((len(self.rho_list), pnum, 3))
        for i, (rho, phi, theta) in enumerate(zip(self.rho_list, self.phi_list, self.theta_list)):
            XList = np.random.choice(np.arange(-xrange, xrange, 0.5), size=pnum)
            YList = np.random.choice(np.arange(-yrange, yrange, 0.5), size=pnum)
            ZList = (- cos(phi) * cos(theta) * XList - cos(phi) * sin(theta) * YList - rho) / sin(phi)

            if biasrange != 0:
                biasX = np.random.choice(np.arange(-biasrange, biasrange, 0.01), pnum)
                biasY = np.random.choice(np.arange(-biasrange, biasrange, 0.01), pnum)
                biasZ = np.random.choice(np.arange(-biasrange, biasrange, 0.01), pnum)

                XList += biasX
                YList += biasY
                ZList += biasZ

            point_clouds[i] = np.vstack((XList, YList, ZList)).T
        
        return point_clouds


if __name__ == "__main__":
    pg = Plane_generator()
    pg.create_planes(5, 20)
    point_clouds = pg.to_pointclouds(50)
    point_clouds = point_clouds.reshape(-1, 3)
    X = point_clouds[:, 0]
    Y = point_clouds[:, 1]
    Z = point_clouds[:, 2]
    plt3d = plt.figure(figsize=(20, 10)).gca(projection='3d')
    plt3d.scatter(X, Y, Z)
    plt3d.set_xlabel("X")
    plt3d.set_ylabel("Y")
    plt3d.set_zlabel("Z")
    plt.show()
