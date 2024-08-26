from data_processing import process_dataset
from visualizer import Visualizer
import os

def main():
    base_dataset_path = '/Users/cmazzoleni/Documents/GitHub/CuboidAbstractionViaSeg/dataset_depth_map/chair'
    origin_pointcloud_folder_path = os.path.join(base_dataset_path, 'segmentation_ply')
    origin_cuboid_folder_path = os.path.join(base_dataset_path, 'cuboid_json')

    # Process the dataset
    process_dataset(base_dataset_path, origin_pointcloud_folder_path, origin_cuboid_folder_path)

if __name__ == "__main__":
    main()
