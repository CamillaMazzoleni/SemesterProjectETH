from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Component:
    scale: List[float]
    rotation: List[float]
    position: List[float]
    epsilon1: float
    epsilon2: float
    color: Tuple[int, int, int] = None  # RGB color

@dataclass
class CuboidData:
    components: List[Component]

class CuboidProcessor:
    def __init__(self, cuboid_data: CuboidData):
        self.cuboid_data = cuboid_data
        self.cuboid_colors = self.extract_cuboid_colors()

    def extract_cuboid_colors(self):
        cuboid_colors = []
        #check if there is a color
        if self.cuboid_data.components[0].color is None:
            return cuboid_colors
        for component in self.cuboid_data.components:
            normalized_color = tuple(np.array(component.color) / 255.0)  # Normalize color to [0, 1] range
            if normalized_color not in cuboid_colors:
                cuboid_colors.append(normalized_color)
        return cuboid_colors

    def get_component_data(self, index: int) -> Component:
        if 0 <= index < len(self.cuboid_data.components):
            return self.cuboid_data.components[index]
        else:
            raise IndexError("Component index out of range.")
    
    def set_component_data(self, index: int, scale=None, rotation=None, position=None, epsilon1=None, epsilon2=None, color=None):
        if 0 <= index < len(self.cuboid_data.components):
            component = self.cuboid_data.components[index]
            if scale is not None:
                component.scale = scale
            if rotation is not None:
                component.rotation = rotation
            if position is not None:
                component.position = position
            if epsilon1 is not None:
                component.epsilon1 = epsilon1
            if epsilon2 is not None:
                component.epsilon2 = epsilon2
            if color is not None:
                component.color = color
        else:
            raise IndexError("Component index out of range.")
