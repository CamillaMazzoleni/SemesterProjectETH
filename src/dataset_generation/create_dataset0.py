import os
import json
import numpy as np
import plyfile
from src.cuboid_generator.CuboidProcessor import CuboidProcessor
from src.utils.json_utils import save_json
from src.visualizers.MayaviVisualizer import MayaviVisualizer
from src.shapenet_processing.PointCloudProcessor import PointCloudProcessor  # Assuming this class has been defined separately
from src.utils.json_utils import load_cuboid_data


def create_entry(shapenet_id, class_id, nr_cubes, component_origins, component_targets):
    return {
        "shapenet_id": shapenet_id,
        "class_id": class_id,
        "nr_cubes": nr_cubes,
        "component_origins": component_origins,
        "component_targets": component_targets
}


def process_point_clouds(segmentation_path, processor, visualizer, pointcloud_views_id_folder_path, base_path):
    plydata = plyfile.PlyData.read(segmentation_path)
    vertex = plydata['vertex']
    points = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
    colors = np.vstack((vertex['red'], vertex['green'], vertex['blue'])).T / 255.0

    point_cloud_colors = set(tuple(color) for color in colors)
    component_targets = [None] * len(processor.cuboid_data.components)

    for color in point_cloud_colors:
        if color not in processor.cuboid_colors:
            continue
        index = processor.cuboid_colors.index(color)
        mask = np.all(colors == color, axis=1)
        segmented_points = points[mask]
        segmented_points -= np.mean(segmented_points, axis=0)
        visualizer.add_pointcloud('pointcloud', segmented_points, color)

        pointcloud_views_component_folder_path = os.path.join(pointcloud_views_id_folder_path, f"component_{index}")
        pointcloud_views = visualizer.save_cube_map_screenshots(pointcloud_views_component_folder_path)
        component_targets[index] = {
            "component_info": {view: os.path.relpath(path, base_path) for view, path in pointcloud_views.items()}
        }
        visualizer.remove_element('pointcloud')

    return component_targets

def visualize_cuboids(processor, visualizer, cuboid_views_id_folder_path, base_path):
    component_origins = []
    for i, component in enumerate(processor.cuboid_data.components):
        visualizer.add_superquadric(
            f"sq_{i}",
            scalings=np.array(component.scale),
            exponents=np.array([component.epsilon1, component.epsilon2]),
            translation=np.array(component.position),
            rotation=np.array(component.rotation),
            color=np.array(component.color)
        )
        
        cubemap_views_component_folder_path = os.path.join(cuboid_views_id_folder_path, f"component_{i}")
        cubemap_views = visualizer.save_cube_map_screenshots(cubemap_views_component_folder_path)
        component_origins.append({
            "component_info": {view: os.path.relpath(path, base_path) for view, path in cubemap_views.items()}
        })
        visualizer.clear_scene()

    return component_origins

def main():
    data = []
    base_path = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/dataset/chair'
    origin_pointcloud_folder_path = os.path.join(base_path, 'segmentation')
    origin_json_folder_path = os.path.join(base_path, 'cuboid_json')
    pointcloud_views_folder_path = os.path.join(base_path, 'pointcloud_views')
    cuboid_views_folder_path = os.path.join(base_path, 'cuboid_views')
    
    segmentation_list = os.listdir(origin_pointcloud_folder_path)
    visualizer = MayaviVisualizer()

    for segmentation in segmentation_list:
        shapenet_id = segmentation.split('_')[0]
        pointcloud_views_id_folder_path = os.path.join(pointcloud_views_folder_path, shapenet_id)
        cuboid_views_id_folder_path = os.path.join(cuboid_views_folder_path, shapenet_id)

        os.makedirs(pointcloud_views_id_folder_path, exist_ok=True)
        os.makedirs(cuboid_views_id_folder_path, exist_ok=True)

        segmentation_path = os.path.join(origin_pointcloud_folder_path, segmentation)
        json_file = os.path.join(origin_json_folder_path, f"{shapenet_id}.json")

        # Load and process cuboid data
        cuboid_data = load_cuboid_data(json_file)
        #center cuboids
        for component in cuboid_data.components:
            component.position = [0, 0, 0]
        cuboid_processor = CuboidProcessor(cuboid_data)

        # Process point clouds
        component_targets = process_point_clouds(segmentation_path, cuboid_processor, visualizer, pointcloud_views_id_folder_path, base_path)

        # Visualize cuboids
        component_origins = visualize_cuboids(cuboid_processor, visualizer, cuboid_views_id_folder_path, base_path)

        class_id = "chair"
        nr_cubes = len(cuboid_processor.cuboid_colors)
        data.append(create_entry(shapenet_id, class_id, nr_cubes, component_origins, component_targets))

    save_json('dataset.json', data)

if __name__ == '__main__':
    main()
