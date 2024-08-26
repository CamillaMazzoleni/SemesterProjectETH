import os
import argparse
from src.utils.dataset_utils import create_directory
from src.visualizers.Open3dVisualizer import Open3DVisualizer
from src.utils.json_utils import load_cuboid_data
from src.utils.plotting_utils import load_mesh_data, load_pointcloud_data
from src.scripts.preprocessing_data.data_to_6views_utils import process_file
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert 3D data (JSON, OBJ, PCD) to six view images.")
    parser.add_argument("input_directory", type=str, help="Directory containing the input .json files.")
    parser.add_argument("output_directory", type=str, help="Directory to save the output images.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Ensure the output directory exists
    create_directory(args.output_directory)

    # Initialize the visualizer
    visualizer = Open3DVisualizer()

    # Iterate over all .json files in the input directory
    for file_name in os.listdir(args.input_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(args.input_directory, file_name)
            file_extension = '.json'  # since we're specifically looking for JSON files
            
            try:
                # Process the input file and generate six views
                process_file(file_path, file_extension, args.output_directory, file_name[:-5], visualizer, save_depth=True)
                print(f"Successfully processed {file_path} and saved views to {args.output_directory}")
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")
            finally:
                visualizer.clear_scene()

    # Close the visualizer when done
    visualizer.close()

if __name__ == "__main__":
    main()