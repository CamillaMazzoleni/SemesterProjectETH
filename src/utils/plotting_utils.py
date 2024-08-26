from PIL import Image
import os
import open3d as o3d

def create_cube_map_2x3(image_paths, output_path, show=False):
    """
    Creates a cube map from six images representing the top, bottom, front, back, left, and right views.

    :param image_paths: A dictionary with keys 'top', 'bottom', 'front', 'back', 'left', and 'right' containing the file paths of the images.
    :param output_path: The file path where the resulting cube map image will be saved.
    """
    # Load the images
    top = Image.open(image_paths['top'])
    bottom = Image.open(image_paths['bottom'])
    front = Image.open(image_paths['front'])
    back = Image.open(image_paths['back'])
    left = Image.open(image_paths['left'])
    right = Image.open(image_paths['right'])

    # Ensure all images are the same size
    width, height = top.size

    # Create an empty image to hold the cube map
    cube_map = Image.new('RGB', (3 * width, 2 * height))

    # Paste images into the cube map
    cube_map.paste(top, (0, 0))
    cube_map.paste(back, (width, 0))
    cube_map.paste(left, (2 * width, 0))
    cube_map.paste(front, (0, height))
    cube_map.paste(right, (width, height))
    cube_map.paste(bottom, (2 * width, height))

    # Save or show the result
    cube_map.save(output_path)
    if show:
        cube_map.show()



def create_cube_map(image_paths, output_path):
    """
    Creates a cube map from six images representing the top, bottom, front, back, left, and right views.

    :param image_paths: A dictionary with keys 'top', 'bottom', 'front', 'back', 'left', and 'right' containing the file paths of the images.
    :param output_path: The file path where the resulting cube map image will be saved.
    """
    # Load the images
    top = Image.open(image_paths['top'])
    bottom = Image.open(image_paths['bottom'])
    front = Image.open(image_paths['front'])
    back = Image.open(image_paths['back'])
    left = Image.open(image_paths['left'])
    right = Image.open(image_paths['right'])

    # Ensure all images are the same size
    width, height = top.size

    # Create a new image with the appropriate size (4x width by 3x height)
    cube_map = Image.new('RGB', (4 * width, 3 * height))

    # Paste images into the cube map
    cube_map.paste(top, (width, 0))
    cube_map.paste(back, (0, height))
    cube_map.paste(left, (width, height))
    cube_map.paste(front, (2 * width, height))
    cube_map.paste(right, (3 * width, height))
    cube_map.paste(bottom, (width, 2 * height))

    # Save or show the result
    cube_map.save(output_path)
    cube_map.show()



def load_mesh_data(file_path):
    try:
        print("Loading mesh data...")
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh.is_empty():
            print("Mesh data is missing vertices or triangles.")
            return None
        print("Mesh data loaded successfully.")
        return mesh
    except Exception as e:
        print(f"An error occurred while loading the mesh data: {e}")
        return None  # O
    

def load_pointcloud_data(file_path):
    return o3d.io.read_point_cloud(file_path)   