import numpy as np
from abc import ABC, abstractmethod

class BaseVisualizer(ABC):
    def __init__(self, window_width=800, window_height=800):
        self.window_width = window_width
        self.window_height = window_height

    @abstractmethod
    def add_superquadric(self, name: str, scalings: np.array, exponents: np.array,
                         translation: np.array, rotation: np.array, color: np.array, resolution: int):
        pass

    

    @abstractmethod
    def save_cube_map_screenshots(self, base_filename: str):
        pass

    @abstractmethod
    def clear_scene(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def show(self):
        pass
