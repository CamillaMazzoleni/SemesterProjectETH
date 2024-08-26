import os
import argparse
from PIL import Image
from src.utils.plotting_utils import create_cube_map_2x3
from src.utils.dataset_utils import rename_file

def process_cube_maps(source_dir, destination_dir, new_suffix):
    """
    Process images in the source directory to create and save cube maps.

    :param source_dir: The directory containing the source images.
    :param destination_dir: The directory to save the cube map images.
    :param new_suffix: The new suffix to rename the output files.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Define the view keywords expected for each part of the cube map
    view_keywords = ['top', 'bottom', 'front', 'back', 'left', 'right']

    # return all the files in the source directory 
    files = os.listdir(source_dir)
    #concatenate the source directory with the file name
    files = [os.path.join(source_dir, file) for file in files]
    

    # Group files by their IDs
    files_grouped_by_id = {}
    for file_path in files:
        file_id = os.path.basename(file_path).split('_')[0]
        if file_id not in files_grouped_by_id:
            files_grouped_by_id[file_id] = {}
        for view in view_keywords:
            if view in file_path:
                files_grouped_by_id[file_id][view] = file_path

    # Process each group to create the cube map
    for file_id, image_paths in files_grouped_by_id.items():
        if len(image_paths) == 6:  # Ensure all six views are present
            output_path = os.path.join(destination_dir, f"{file_id}_temp.png")
            create_cube_map_2x3(image_paths, output_path)

            # Rename the file
            renamed_path = rename_file(output_path, new_suffix)
            print(f"Cube map saved as {renamed_path}")
        else:
            print(f"Skipping {file_id} due to missing views")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Create cube maps from a set of images.")
    parser.add_argument('--source_dir', type=str, required=True, help="Directory containing source images")
    parser.add_argument('--destination_dir', type=str, required=True, help="Directory to save cube map images")
    parser.add_argument('--new_suffix', type=str, required=True, help="Suffix to append to the renamed files")

    args = parser.parse_args()

    # Process the cube maps
    process_cube_maps(args.source_dir, args.destination_dir, args.new_suffix)
