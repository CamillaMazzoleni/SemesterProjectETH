import json
from src.cuboid_generator.CuboidProcessor import CuboidData, Component
import open3d as o3d

def save_json(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def remove_prefix_from_paths(data, prefix):
    if isinstance(data, dict):
        return {key: remove_prefix_from_paths(value, prefix) for key, value in data.items()}
    elif isinstance(data, list):
        return [remove_prefix_from_paths(item, prefix) for item in data]
    elif isinstance(data, str) and data.startswith(prefix):
        return data[len(prefix):]
    else:
        return data

def load_cuboid_data(json_file_path):
    with open(json_file_path, 'r') as file:
        cuboid_data_dict = json.load(file)
    
    components = []
    for component in cuboid_data_dict["components"]:
        # Check if the color is present; if not, assign a default color
        if "color" not in component:
            component["color"] = [0, 255, 0]  # Default color (Green)
        else:
            component["color"] = [0, 255, 0]

        # Create a Component instance for each component in the JSON
        components.append(Component(
            scale=component["scale"],
            rotation=component["rotation"],
            position=component["position"],
            epsilon1=component["epsilon1"],
            epsilon2=component["epsilon2"],
            color=tuple(component["color"])  # Convert list to tuple
        ))

    return CuboidData(components=components)




