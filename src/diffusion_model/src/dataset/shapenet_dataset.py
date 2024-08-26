import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class ShapeNetDataset(Dataset):
    def __init__(self, json_path, image_base_dir, transform=None, depth_transform=None, resize_to_hw=None):
        """
        Args:
            json_path (str): Path to the JSON file containing the dataset information.
            image_base_dir (str): Base directory where the images are stored.
            transform (callable, optional): Optional transform to be applied on an RGB image.
            depth_transform (callable, optional): Optional transform to be applied on a depth map.
            resize_to_hw (tuple, optional): Target size to resize the images (height, width).
        """
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.depth_transform = depth_transform
        self.resize_to_hw = resize_to_hw

        # Load the dataset JSON file
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the data item
        item = self.data[idx]
        
        # Load the images
        complete_view_path = os.path.join(self.image_base_dir, item['complete_view'])
        cuboid_depth_path = os.path.join(self.image_base_dir, item['cuboid_depth'])
        mesh_depth_path = os.path.join(self.image_base_dir, item['mesh_depth'])
        
        complete_view_image = Image.open(complete_view_path).convert('RGB')
        cuboid_depth_image = Image.open(cuboid_depth_path)
        mesh_depth_image = Image.open(mesh_depth_path)
        
        # Apply resizing if specified
        if self.resize_to_hw:
            complete_view_image = complete_view_image.resize(self.resize_to_hw, Image.BILINEAR)
            cuboid_depth_image = cuboid_depth_image.resize(self.resize_to_hw, Image.NEAREST)
            mesh_depth_image = mesh_depth_image.resize(self.resize_to_hw, Image.NEAREST)
        
        # Convert to numpy arrays
        complete_view_image = np.array(complete_view_image)
        cuboid_depth_image = np.array(cuboid_depth_image)
        mesh_depth_image = np.array(mesh_depth_image)
        
        # Normalize RGB images if a transform is provided
        if self.transform:
            complete_view_image = self.transform(complete_view_image)

        # Normalize depth maps if a transform is provided
        if self.depth_transform:
            cuboid_depth_image = self.depth_transform(cuboid_depth_image)
            mesh_depth_image = self.depth_transform(mesh_depth_image)
        else:
            cuboid_depth_image = torch.from_numpy(cuboid_depth_image).float()
            mesh_depth_image = torch.from_numpy(mesh_depth_image).float()
        
        # Package the data into a dictionary
        sample = {
            'shape_id': item['shape_id'],
            'complete_view': complete_view_image,
            'cuboid_depth': cuboid_depth_image,
            'mesh_depth': mesh_depth_image
        }
        
        return sample
