from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt
from src.superquadric_difformation.superquadric_class import SuperQuadrics
from src.visualizers.BaseVisualizer import BaseVisualizer
from src.utils.emsfitting_visualization_utils import uniformSampledSuperellipse

class MayaviVisualizer(BaseVisualizer):
    def __init__(self, window_width=800, window_height=800):
        super().__init__(window_width, window_height)
        self.elements = {}
        self.fig = mlab.figure(size=(window_width, window_height), bgcolor=(1, 1, 1))

    def add_superquadric(self, name: str, scalings: np.array, exponents: np.array,
                         translation: np.array, rotation: np.array, color: np.array, resolution: int = 100):
        """Adds a superquadric mesh to the scene."""

        # Initialize the SuperQuadrics class
        sq = SuperQuadrics(scalings, exponents, resolution)
        vertices = np.stack((sq.x.flatten(), sq.y.flatten(), sq.z.flatten()), axis=-1)

        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                triangles.append([i * resolution + j, (i + 1) * resolution + j + 1, (i + 1) * resolution + j])
                triangles.append([i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j + 1])

        vertices = np.array(vertices)
        triangles = np.array(triangles)

        # Apply color
        color = color / 255.0  # Normalize the color

        # Create a Mayavi mesh
        mesh = mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles,
                                    color=tuple(color), representation='surface')

        # Apply rotation and translation
        rotated_vertices = np.dot(vertices, rotation.T)
        mesh.mlab_source.set(x=rotated_vertices[:, 0] + translation[0],
                             y=rotated_vertices[:, 1] + translation[1],
                             z=rotated_vertices[:, 2] + translation[2])
        
        self.elements[name] = mesh

        return mesh, vertices
    

    def showSuperquadrics_ems_fitting(self, name: str, shape, scale, rotation, translation, color, threshold = 1e-2, num_limit = 10000, arclength = 0.02):
        # avoid numerical insftability in sampling
        if shape[0] < 0.007:
            shape[0] = 0.007
        if shape[1] < 0.007:
            shape[1] = 0.007
        # sampling points in superellipse    
        point_eta = uniformSampledSuperellipse(shape[0], [1, scale[2]], threshold, num_limit, arclength)
        point_omega = uniformSampledSuperellipse(shape[1], [scale[0], scale[1]], threshold, num_limit, arclength)
        
        # preallocate meshgrid
        x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
        y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
        z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))

        for m in range(np.shape(point_omega)[1]):
            for n in range(np.shape(point_eta)[1]):
                point_temp = np.zeros(3)
                point_temp[0 : 2] = point_omega[:, m] * point_eta[0, n]
                point_temp[2] = point_eta[1, n]
                point_temp = rotation @ point_temp + translation

                x_mesh[m, n] = point_temp[0]
                y_mesh[m, n] = point_temp[1]
                z_mesh[m, n] = point_temp[2]
        
        #add mesh to the elements dictionary
        mesh = mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=1)
        self.elements[name] = mesh
        
        mlab.view(azimuth=0.0, elevation=0.0, distance=2)
        mlab.mesh(x_mesh, y_mesh, z_mesh, color=color, opacity=1)

    def save_cube_map_screenshots(self, base_filename: str, distance: float = 1):
        views = [
            (0, 0),   # Front
            (180, 0), # Back
            (90, 0),  # Left
            (-90, 0), # Right
            (0, 90),  # Top
            (0, -90)  # Bottom
        ]
        view_name = ['front', 'back', 'left', 'right', 'top', 'bottom']
        filenames = {}
        
        for i, (azimuth, elevation) in enumerate(views):
            filename = f"{base_filename}_view_{view_name[i]}.png"
            self.save_screenshot(filename, azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=(0, 0, 0))
            filenames[view_name[i]] = filename
        return filenames
    

    def add_pointcloud(self, name: str, points, colors, visible: bool = True):
        point_cloud = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=colors, scale_factor=0.01)
        self.elements[name] = point_cloud

    def clear_scene(self):
        mlab.clf(self.fig)

    def close(self):
        mlab.close(self.fig)

    def show(self):
        mlab.show()
    
    def save_screenshot(self, filename: str, azimuth: float = 0, elevation: float = 0, distance: float = None, focalpoint: tuple = (0, 0, 0)):
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint, figure=self.fig)
        screenshot = mlab.screenshot(figure=self.fig, mode='rgb', antialiased=True)
        plt.imsave(filename, screenshot)
        return screenshot
    
    def remove_element(self, name: str):
        self.elements[name].remove()
        del self.elements[name]
