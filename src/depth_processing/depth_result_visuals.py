import os
import matplotlib.pyplot as plt
from PIL import Image

import os
import re

def get_prefixes(folder_path, n=10):
    """
    Extracts the first `n` unique prefixes from the files in the given folder.

    Args:
    folder_path (str): The path to the folder containing the images.
    n (int): The number of unique prefixes to extract.

    Returns:
    list: A list of the first `n` unique prefixes.
    """
    filenames = os.listdir(folder_path)
    prefixes = set()

    for filename in filenames:
        # Extract the prefix using regex or simple split
        prefix_match = re.match(r'(.+)_view_', filename)
        if prefix_match:
            prefix = prefix_match.group(1)
            prefixes.add(prefix)
            if len(prefixes) >= n:
                break

    return list(prefixes)


def load_image_with_prefix(folder_path, prefix, view_type):
    """
    Loads an image with a specific prefix and view type from a folder.

    Args:
    folder_path (str): The path to the folder containing the image.
    prefix (str): The common prefix of the image file.
    view_type (str): The view type of the image file (e.g., '_view_back').

    Returns:
    Image: The loaded image.
    """
    filename = f"{prefix}{view_type}.png"  # Assuming the image format is PNG
    image_path = os.path.join(folder_path, filename)
    return Image.open(image_path)

def display_and_save_images(base_path, prefix, view_type, output_path):
    """
    Loads, displays, and saves three images with the same prefix and view type from different folders side by side.

    Args:
    base_path (str): The root path where the folders are located.
    prefix (str): The common prefix of the images.
    view_type (str): The view type of the images (e.g., '_view_back').
    output_path (str): The path to save the output image.
    """
    folder_suffixes = ['chair_6views', 'chair_6views_depths_anything', 'chair_6views_depths_marigold']
    image_suffixes = [view_type, f"{view_type}_depth", f"{view_type}_depth_16bit"]
    
    images = [
        load_image_with_prefix(os.path.join(base_path, folder_suffix), prefix, image_suffix)
        for folder_suffix, image_suffix in zip(folder_suffixes, image_suffixes)
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["Original", "Depth (Anything)", "Depth (Marigold)"]

    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()

# Example usage
folder_path = '../data/processed/chair_6views'  # Adjust this to the folder you want to scan
prefixes = get_prefixes(folder_path, 10)
base_path = '../data/processed'
view_types = ['_view_back' , '_view_top', '_view_front', '_view_bottom', '_view_right','_view_left' ] # The view type (e.g., '_view_back', '_view_front')
output_path = '../results/depth_processing'

for prefix in prefixes:
    for view_type in view_types:
        output_path = os.path.join(output_path, f'{prefix}{view_type}_comparison.png')
        display_and_save_images(base_path, prefix, view_type, output_path)

