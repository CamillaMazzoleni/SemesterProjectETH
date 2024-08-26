import os
import numpy as np
import plyfile
from plyfile import PlyData 





class PointCloudProcessor:
    def __init__(self, file_path: str):
        self.points, self.colors = self.load_point_cloud(file_path)

    def load_point_cloud(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:

        plydata = plyfile.PlyData.read(file_path)
        plydata = plydata['vertex']
        points = self.extract_points(plydata)
        colors = self.extract_colors(plydata)
        return points, colors

    def extract_points(self, plydata: plyfile.PlyData) -> np.ndarray:
        return np.vstack((plydata['x'], plydata['y'], plydata['z'])).T

    def extract_colors(self, plydata: plyfile.PlyData) -> np.ndarray:
        return np.vstack((plydata['red'], plydata['green'], plydata['blue'])).T / 255.0

    def segment_points_by_color(self, color: tuple) -> np.ndarray:
        mask = np.all(self.colors == color, axis=1)
        segmented_points = self.points[mask]
        return segmented_points - np.mean(segmented_points, axis=0)

    def process_point_clouds(self, cuboid_colors: list) -> list:
        return [self.segment_points_by_color(color) for color in cuboid_colors if color in set(tuple(c) for c in self.colors)]
    


