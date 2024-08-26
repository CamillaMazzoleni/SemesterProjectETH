import os
import json
import numpy as np
from PIL import Image
from src.visualizers.MayaviVisualizer import MayaviVisualizer
from src.utils.json_utils import load_json
from src.utils.plotting_utils import create_cube_map_2x3

def process_component_origins(component_origins, base_path, output_directory, shapenet_id):
    processed_origins = []
    for idx, component in enumerate(component_origins):
        component_info = component['component_info']
        
        # Prepare image paths dictionary for create_cube_map_2x3 function
        image_paths = {
            'top': os.path.join(base_path, component_info['top']),
            'bottom': os.path.join(base_path, component_info['bottom']),
            'front': os.path.join(base_path, component_info['front']),
            'back': os.path.join(base_path, component_info['back']),
            'left': os.path.join(base_path, component_info['left']),
            'right': os.path.join(base_path, component_info['right'])
        }
        
        output_path = os.path.join(output_directory, f'{shapenet_id}_origin_component_{idx}_cube_map.jpg')
        
        # Use create_cube_map_2x3 to create the cube map image
        create_cube_map_2x3(image_paths, output_path)
        
        processed_origins.append({"cube_map": os.path.basename(output_path)})
    return processed_origins

def process_component_targets(component_targets, base_path, output_directory, shapenet_id):
    processed_targets = []
    for idx, component in enumerate(component_targets):
        component_info = component['component_info']
        
        # Prepare image paths dictionary for create_cube_map_2x3 function
        image_paths = {
            'top': os.path.join(base_path, component_info['top']),
            'bottom': os.path.join(base_path, component_info['bottom']),
            'front': os.path.join(base_path, component_info['front']),
            'back': os.path.join(base_path, component_info['back']),
            'left': os.path.join(base_path, component_info['left']),
            'right': os.path.join(base_path, component_info['right'])
        }
        
        output_path = os.path.join(output_directory, f'{shapenet_id}_target_component_{idx}_cube_map.jpg')
        
        # Use create_cube_map_2x3 to create the cube map image
        create_cube_map_2x3(image_paths, output_path)
        
        processed_targets.append({"cube_map": os.path.basename(output_path)})
    return processed_targets

def main():
    base_path = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/dataset/chair'
    output_directory = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/dataset_cubemap'
    os.makedirs(output_directory, exist_ok=True)

    dataset_file = 'dataset.json'
    dataset = load_json(dataset_file)

    new_dataset = []

    for item in dataset:
        shapenet_id = item['shapenet_id']
        new_item = {
            "shapenet_id": shapenet_id,
            "class_id": item['class_id'],
            "nr_cubes": item['nr_cubes'],
            "complete_cuboid": "",
            "component_origins": [],
            "component_targets": []
        }

        cuboid_json_path = os.path.join(base_path, 'cuboid_json', f"{shapenet_id}.json")
        json_data = load_json(cuboid_json_path)

        visualizer = MayaviVisualizer()

        for i, component in enumerate(json_data["components"]):
            visualizer.add_superquadric(
                f"sq_{i}",
                scalings=np.array(component["scale"]),
                exponents=np.array([component["epsilon1"], component["epsilon2"]]),
                translation=np.array(component["position"]),
                rotation=np.array(component["rotation"]),
                color=np.array(component["color"])
            )

        complete_screenshot_path = os.path.join(output_directory, f"{shapenet_id}_complete_object.png")
        visualizer.save_screenshot(complete_screenshot_path)
        new_item['complete_cuboid'] = os.path.basename(complete_screenshot_path)

        new_item['component_origins'] = process_component_origins(item['component_origins'], base_path, output_directory, shapenet_id)
        new_item['component_targets'] = process_component_targets(item['component_targets'], base_path, output_directory, shapenet_id)

        new_dataset.append(new_item)
        visualizer.clear_scene()

    with open(os.path.join(output_directory, 'new_dataset.json'), 'w') as file:
        json.dump(new_dataset, file, indent=4)

    print("Dataset processing complete. New dataset saved as 'new_dataset.json'.")

if __name__ == '__main__':
    main()

