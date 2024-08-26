import os
import shutil

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_item(item_type, **kwargs):
    """Creates a dictionary entry for the dataset with flexible parameters."""
    
    # Default fields
    default_fields = {
        "shapenet_id": "",
        "class_id": "",
        "nr_cubes": 0,
        "complete_cuboid": "",
        "component_origins": [],
        "component_targets": [],
        "depth_view_camera_params": {},
        "cuboid_cubemaps": {},
        "depth_cubemaps": {}
    }
    
    # Specific initialization based on item_type
    if item_type == "new_item":
        default_fields["class_id"] = "chair"
        default_fields.update(kwargs)  # Update with any additional parameters

    elif item_type == "entry":
        # Remove fields not relevant to an entry
        for key in ["complete_cuboid", "depth_view_camera_params", "cuboid_cubemaps", "depth_cubemaps"]:
            default_fields.pop(key)
        default_fields.update(kwargs)

    # Overwrite default fields with provided keyword arguments
    return default_fields

def move_file(file_path, destination_dir):
    """
    Move a file from its current location to a destination directory.

    Parameters:
    - file_path: str, the full path to the file to be moved.
    - destination_dir: str, the path to the destination directory.
    """
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Move the file
    shutil.move(file_path, destination_dir)
    print(f"Moved {file_path} to {destination_dir}")


def rename_file(file_path, new_suffix):
    """
    Rename a file based on its ID and a new suffix.

    Parameters:
    - file_path: str, the full path to the file to be renamed.
    - new_suffix: str, the new suffix to append to the file ID.
    """
    # Extract the ID from the filename (everything before the first underscore)
    directory, original_filename = os.path.split(file_path)
    file_id = os.path.splitext(original_filename)[0].split('_')[0]
    
    # Define the new file name and full path
    new_filename = f"{file_id}_{new_suffix}.png"
    new_file_path = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(file_path, new_file_path)
    print(f"Renamed {file_path} to {new_file_path}")
    
    return new_file_path

def filter_files_by_keyword(source_dir, search_keywords):
    """
    Filter files in a directory based on all keywords.

    Parameters:
    - source_dir: str, the path to the source directory.
    - search_keywords: list of str, the keywords that must all be present in file names.

    Returns:
    - List of file paths that match all of the search keywords.
    """
    matching_files = []
    for filename in os.listdir(source_dir):
        if all(keyword in filename for keyword in search_keywords):
            matching_files.append(os.path.join(source_dir, filename))
    return matching_files


def save_to_txt(input_folder, output_file):
    # Get the list of directories/files
    items = os.listdir(input_folder)
    
    # Save the list to a TXT file
    with open(output_file, 'w') as f:
        for item in items:
            f.write(f"{item}\n")  # Write each item on a new line

def read_from_txt(file_path):
    with open(file_path, 'r') as f:
        items = f.read().splitlines()  # Read all lines and remove newline characters
    return items
