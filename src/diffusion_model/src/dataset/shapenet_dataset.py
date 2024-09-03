import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode
from typing import Union
from src.util.depth_transform import DepthNormalizerBase


class ShapeNetDataset(Dataset):
    def __init__(
        self, 
        json_path: str,
        image_base_dir: str,
        transform=None,
        depth_transform=None,
        resize_to_hw=(200,200),
        **kwargs,
    ):
        self.image_base_dir = image_base_dir
        self.resize_to_hw = resize_to_hw

        # Load the dataset JSON file
        with open(json_path, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        # Use BaseDepthDataset length if JSON is not used
        return super().__len__() if not hasattr(self, 'data') else len(self.data)

    def __getitem__(self, idx):
        # If JSON is used, load data from JSON and use BaseDepthDataset methods
        if hasattr(self, 'data'):
            item = self.data[idx]
            complete_view_image = self.load_rgb_image(item['complete_view'])
            
            cuboid_depth_image = self.load_depth_image(item['cuboid_depth'])
            mesh_depth_image = self.load_depth_image(item['mesh_depth'])

            """
            cuboid_depth_image = self.load_rgb_image(item['cuboid_depth'])
            mesh_depth_image = self.load_rgb_image(item['mesh_depth'])
            """
            
            sample = {
                'shape_id': item['shape_id'],
                'complete_view': complete_view_image,
                'cuboid_depth': cuboid_depth_image,
                'mesh_depth': mesh_depth_image
            }

            return sample
        
    
    def load_rgb_image(self, path):
        #get path
        full_path = os.path.join(self.image_base_dir, path)
        #read image
        input_image = Image.open(full_path).convert('RGB')
        input_image = input_image.resize(self.resize_to_hw)
        image = np.asarray(input_image)
        
        #read rgb file
        rgb = np.transpose(image, (2, 0, 1)).astype(int)
        
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).float()
        return rgb_norm

    def load_depth_image(self, path):
        #get path
        full_path = os.path.join(self.image_base_dir, path)
        #read image

        depth_raw = Image.open(full_path).convert('L')
        depth_raw = depth_raw.resize(self.resize_to_hw)
        depth_raw = np.asarray(depth_raw)
        depth_raw = depth_raw.copy()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        return depth_raw_linear

