import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from generate_planes import Plane_generator

class hough(object):
    def __init__(self, theta_step, phi_step, rho_step):
        """
        Param:
        theta_step: The step of sampling theta
        phi_step: The step of sampling phi
        rho_step:The step of sampling rho
        """
        self.pointListX = None
        self.pointListY = None
        self.pointListZ = None

        self.theta = np.arange(0, 360, theta_step)
        self.phi = np.arange(-90, 90, phi_step)
        self.rho = None

        self.rho_step = rho_step

        self.n_theta = len(self.theta)
        self.n_phi = len(self.phi)
        self.n_rho = None

        self.H = None
        
        self.pg = None

        return

    def scan_points(self, points_path, sub_sample=None):
        """
        Loading point clouds from a .mat file, the read mat file is m
        
        Params:
        points_path: The path of the .mat file
        sub_sample: Just in case the number of points is too much, you can use this to subsample by the number given
        """
        m = loadmat(points_path)
        # The data is stored in this variable, can be modified
        XYZ = m['XYZcamera']

        if sub_sample is None:
            self.pointListX = XYZ[:, :, 0].reshape(-1, )
            self.pointListY = XYZ[:, :, 1].reshape(-1, )
            self.pointListZ = XYZ[:, :, 2].reshape(-1, )
            return
        else:
            self.pointListX = XYZ[:, :, 0].reshape(-1, )[::sub_sample]
            self.pointListY = XYZ[:, :, 1].reshape(-1, )[::sub_sample]
            self.pointListZ = XYZ[:, :, 2].reshape(-1, )[::sub_sample]
            return
    
    def load_from_PG(self, plane_num=4, pnum=100, rho_range=5, xrange=50, yrange=50, biasrange=0.):
        """
        Using the plane generator to generate random planes
        
        Params:
        plane_num: The number of planes to be generated
        pnum: The number of points in a certain plane to be generated
        rho_range: The maximum possible value of rho
        x/yrange: The maximum absolute value of X and Y
        biasrange: The maximum absolute value of bias added to each point, set to 0 by default
        """
        self.pg = Plane_generator()
        self.pg.create_planes(plane_num=plane_num, rho_range=rho_range)
        pointclouds = self.pg.to_pointclouds(pnum=pnum, xrange=xrange, yrange=yrange, biasrange=biasrange)
        pointclouds = pointclouds.reshape(-1, 3)
        
        self.pointListX = pointclouds[:, 0]
        self.pointListY = pointclouds[:, 1]
        self.pointListZ = pointclouds[:, 2]
        return

    def load_from_obj(self, obj_path):
        """
        Loading point clouds from obj file 
        (OBJ File for LoD2 meshes in Helsinki has discrete dots, so the plane detection is not working very well)

        """
        obj_file = open(obj_path, 'r')
        lines = np.array([ line.split()[1:4] for line in obj_file.readlines() if line.startswith('v') ]).astype(np.float)
        mean = np.mean(lines, axis=0)
        vertices = lines - mean
        
        self.pointListX = vertices[:, 0]
        self.pointListY = vertices[:, 1]
        self.pointListZ = vertices[:, 2]

        return
        
        

    def SHT(self):
        """
        Run Hough Transformation 
        and store the accumulate result of each plane in H
        H has a shape of (n_theta, n_phi, n_rho)
        """

        X_min = np.min(np.abs(self.pointListX))
        Y_min = np.min(np.abs(self.pointListY))
        Z_min = np.min(np.abs(self.pointListZ))
        X_max = np.max(np.abs(self.pointListX))
        Y_max = np.max(np.abs(self.pointListY))
        Z_max = np.max(np.abs(self.pointListZ))

        # Calculate the maximum distance from original and use it to sample rho
        dis_min = np.sqrt(X_min ** 2 + Y_min ** 2 + Z_min ** 2)
        dis_max = np.sqrt(X_max ** 2 + Y_max ** 2 + Z_max ** 2)

        self.rho = np.arange(-dis_max, dis_max, self.rho_step)
        self.n_rho = len(self.rho)

        # Create theta and pho matrix in radian for multiplication
        theta_mat = np.array([ self.theta for i in range(self.n_phi) ]) * np.pi / 180.
        phi_mat = np.array([ self.phi for i in range(self.n_theta) ]).T * np.pi / 180.

        self.H = np.zeros((self.n_theta, self.n_phi, self.n_rho))
        ratio = (self.n_rho - 1) / (self.rho[-1] - self.rho[0])
        
        for k in range(len(self.pointListX)):
            rho_mat = np.cos(phi_mat) * np.cos(theta_mat) * self.pointListX[k] + np.cos(phi_mat) * np.sin(theta_mat) * self.pointListY[k] + np.sin(phi_mat) * self.pointListZ[k]
            rho_index = np.floor(ratio * (rho_mat - self.rho[0])).astype(np.int32)
            for i in range(self.n_phi):
                for j in range(self.n_theta):
                    self.H[j, i, rho_index[i, j]] += 1 

    
    def plane_extraction(self, threshold=None, x_grid_range=None, y_grid_range=None, figsize=None, elev=30, azim=30):
        """
        Extract the plane according to H and plot detected planes and dots

        Params:
        threshold: The theshold of accumulate value to be consider as a plane; if not given, then consider only the top 3 value
        x/y_grid_range: The grid range for plotting the plane and dots
        figsize: Size of figure
        elev & azim: parameters for rotation of the initial display of the plotted 3D figure
        """
        if threshold is None:
            threshold = (-np.sort(-np.unique(self.H)))[1]
            print((-np.sort(-np.unique(self.H))))        

        if x_grid_range is None:
            x_grid_range = np.arange(-100, 100, 20)
        else:
            x_grid_range = np.arange(-x_grid_range, x_grid_range, 5)
        
        if y_grid_range is None:
            y_grid_range = np.arange(-100, 100, 20)
        else:
            y_grid_range = np.arange(-y_grid_range, y_grid_range, 5)
        
        if figsize is None:
            figsize = (20, 20)


        # gt_theta, gt_phi, gt_rho = self.pg.get_planes()

        detected_plane_index = np.where(self.H > threshold)
        theta_indices = detected_plane_index[0]
        phi_indices = detected_plane_index[1]
        rho_indices = detected_plane_index[2]

        print("%d detected planes." % len(theta_indices))
        plt3d = plt.figure(figsize=figsize).gca(projection='3d')
        plt3d.scatter(self.pointListX, self.pointListY, self.pointListZ, c='red')
        plt3d.view_init(elev, azim)
        plt3d.set_xlabel("X")
        plt3d.set_ylabel("Y")
        plt3d.set_zlabel("Z")
        
        for theta_index, phi_index, rho_index in zip(theta_indices, phi_indices, rho_indices):
            theta = self.theta[theta_index] * np.pi / 180.
            phi = self.phi[phi_index] * np.pi / 180.
            rho = self.rho[rho_index]

            A, B, C, D = np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi), -rho

            if (abs(C - 0.)) > 1e-6:
                # deal with vertical planes
                xx, yy = np.meshgrid(x_grid_range, y_grid_range)
                zz = (-A * xx - B * yy - D) * 1. / C
             
            elif (abs(B - 0.)) > 1e-6:
                xx, zz = np.meshgrid(x_grid_range, y_grid_range)
                yy = (-A * xx - C * zz - D) * 1. / B
            
            elif (abs(A - 0.)) > 1e-6:
                yy, zz = np.meshgrid(x_grid_range, y_grid_range)
                xx = (-B * yy - C * zz - D) * 1. / A

            plt3d.plot_surface(xx, yy, zz, alpha=0.5)
            
        plt.show()
        return

if __name__ == "__main__":
    HT = hough(30, 10, 1)
    # HT.scan_points('./XYZcamera.mat', 100)
    # HT.load_from_PG(plane_num=5, pnum=80, biasrange=0, rho_range=10, xrange=10, yrange=10)
    HT.load_from_obj('B1.obj')
    HT.SHT()
    HT.plane_extraction(figsize=(10, 10), x_grid_range=15, y_grid_range=15)

