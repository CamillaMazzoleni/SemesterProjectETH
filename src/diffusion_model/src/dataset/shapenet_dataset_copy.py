import os
import io
import random
from enum import Enum
from typing import Union
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode, DatasetMode
from src.util.depth_transform import DepthNormalizerBase
import matplotlib.pyplot as plt
from PIL import Image  # For reading PNG images
from scipy.ndimage import binary_dilation
from torchvision.transforms import InterpolationMode, Resize
import open3d as o3d
import cv2


import matplotlib.pyplot as plt

def visualize_mask(mask, title="Mask"):
    """Visualize a given mask tensor and save it as an image."""
    plt.imshow(mask.squeeze(0), cmap='gray')  # Assuming the mask is [1, H, W] or [H, W]
    plt.title(title)
    plt.axis('off')  # Hide axes for clarity
    plt.savefig("example.png")  # Use savefig instead of save
    plt.show()


class ShapeNetDataset(BaseDepthDataset):
    def __init__(
        self,
        mode: DatasetMode,
        json_path: str,
        image_base_dir: str,
        disp_name: str,
        name_mode: DepthFileNameMode = 1,
        resize_to_hw=(20, 20),
        min_depth: float = 0.0,
        max_depth: float = 255.0,
        **kwargs,

    ):
        # Call BaseDepthDataset initializer
        super().__init__(
            mode=mode,
            filename_ls_path=json_path,
            dataset_dir=image_base_dir,
            disp_name=disp_name,
            min_depth=min_depth,
            max_depth=max_depth,
            has_filled_depth=False,  # Assuming filled depth is used
            resize_to_hw=resize_to_hw,
            name_mode=name_mode,
            **kwargs
        )
        
        # Load the dataset JSON file
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        # Use BaseDepthDataset length if JSON is not used
        return super().__len__() if not hasattr(self, 'data') else len(self.data)

    

def save_normalized_depth_maps(self, rasters):
    metadata = {}

    # Step 1 - Save cuboid depth raw normalized image
    cuboid_depth_raw = rasters["cuboid_depth"]["depth_raw_norm"].cpu().numpy()
    cuboid_depth_raw_path = "cuboid_depth_raw_norm.png"
    cv2.imwrite(cuboid_depth_raw_path, cuboid_depth_raw)
    
    metadata['cuboid_depth_raw_norm'] = {
        "description": "Cuboid Depth Raw Normalized",
        "dtype": str(cuboid_depth_raw.dtype),
        "min_value": float(np.min(cuboid_depth_raw)),
        "max_value": float(np.max(cuboid_depth_raw)),
        "path_to_image": cuboid_depth_raw_path
    }

    # Step 2 - Save cuboid depth filled normalized image
    cuboid_depth_filled = rasters["cuboid_depth"]["depth_filled_norm"].cpu().numpy()
    cuboid_depth_filled_path = "cuboid_depth_filled_norm.png"
    cv2.imwrite(cuboid_depth_filled_path, cuboid_depth_filled)
    
    metadata['cuboid_depth_filled_norm'] = {
        "description": "Cuboid Depth Filled Normalized",
        "dtype": str(cuboid_depth_filled.dtype),
        "min_value": float(np.min(cuboid_depth_filled)),
        "max_value": float(np.max(cuboid_depth_filled)),
        "path_to_image": cuboid_depth_filled_path
    }

    # Step 3 - Save mesh depth raw normalized image
    mesh_depth_raw = rasters["mesh_depth"]["depth_raw_norm"].cpu().numpy()
    mesh_depth_raw_path = "mesh_depth_raw_norm.png"
    cv2.imwrite(mesh_depth_raw_path, mesh_depth_raw)
    
    metadata['mesh_depth_raw_norm'] = {
        "description": "Mesh Depth Raw Normalized",
        "dtype": str(mesh_depth_raw.dtype),
        "min_value": float(np.min(mesh_depth_raw)),
        "max_value": float(np.max(mesh_depth_raw)),
        "path_to_image": mesh_depth_raw_path
    }

    # Step 4 - Save mesh depth filled normalized image
    mesh_depth_filled = rasters["mesh_depth"]["depth_filled_norm"].cpu().numpy()
    mesh_depth_filled_path = "mesh_depth_filled_norm.png"
    cv2.imwrite(mesh_depth_filled_path, mesh_depth_filled)
    
    metadata['mesh_depth_filled_norm'] = {
        "description": "Mesh Depth Filled Normalized",
        "dtype": str(mesh_depth_filled.dtype),
        "min_value": float(np.min(mesh_depth_filled)),
        "max_value": float(np.max(mesh_depth_filled)),
        "path_to_image": mesh_depth_filled_path
    }

    # Save the metadata to a JSON file
    with open("depth_images_metadata.json", 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    print("Images saved and metadata written to 'depth_images_metadata.json'")



    def __getitem__(self, idx):
        # Fetch item paths from self.data as previously done
        item = self.data[idx]
        complete_view_image = super()._load_rgb_data(item['complete_view'])
        cuboid_depth_image = super()._load_depth_data(item['cuboid_depth'],item['cuboid_depth'] )
        mesh_depth_image = super()._load_depth_data(item['mesh_depth'], item['mesh_depth'])

        rasters = {
            "complete_view": complete_view_image,
            "cuboid_depth": cuboid_depth_image,
            "mesh_depth": mesh_depth_image,
        }

        # Generate valid masks using the BaseDepthDataset method
        rasters["valid_mask_raw"] = self._get_valid_mask(rasters["cuboid_depth"]['depth_raw_linear'])
        rasters["valid_mask_filled"] = self._get_valid_mask(rasters["mesh_depth"]['depth_raw_linear'])
        rasters["path_to_image"] = item['complete_view']

        # Visualize masks (for debugging purposes)
        visualize_mask(rasters["valid_mask_raw"], title="Raw Valid Mask")
        visualize_mask(rasters["valid_mask_filled"], title="Filled Valid Mask")
        other = {"index": idx, "rgb_relative_path": item['complete_view']}
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        outputs = rasters
        outputs.update(other)
        # Example function call
        self.save_normalized_depth_maps(rasters)
        return outputs


    def _read_depth_file(self, rel_path):
        metadata = {}
        image_to_read = os.path.join(self.dataset_dir, rel_path)
        depth_image = o3d.io.read_image(image_to_read)
        #step 1 - save image
        depth_array = np.asarray(depth_image)
        png_save_path_step1 = "depth_array_raw_opencv.png"
        metadata['step1'] = {
            "description": "Step1 - Input depth image (in mm)",
            "dtype": str(depth_array.dtype),
            "min_value": float(np.min(depth_array)),
            "max_value": float(np.max(depth_array)),
            "path_to_image" : png_save_path_step1
        }
        
        cv2.imwrite(png_save_path_step1, depth_array)
        depth_decoded = depth_array / 1000.0
        #step 2 - save image
        png_save_path_step2 = "depth_decoded_raw_opencv.png"
        metadata['step2'] = {
            "description": "Step 2 - Depth image (in meters)",
            "dtype": str(depth_decoded.dtype),
            "min_value": float(np.min(depth_decoded)),
            "max_value": float(np.max(depth_decoded)),
            "path_to_image" : png_save_path_step2
        }
        cv2.imwrite(png_save_path_step2, depth_decoded)
        with open("depth_images_reading.json", 'w') as json_file:
            json.dump(metadata, json_file, indent=4)

        print("Images saved and metadata written to 'depth_images_reading.json'")
        return depth_decoded

    def _training_preprocess(self, rasters):
        # Augmentation
        """
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)
        """

        # Normalization
        rasters["cuboid_depth"]["depth_raw_norm"] = self.depth_transform(
            rasters["cuboid_depth"]["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["cuboid_depth"]["depth_filled_norm"] = self.depth_transform(
            rasters["cuboid_depth"]["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()

        # Normalization
        rasters["mesh_depth"]["depth_raw_norm"] = self.depth_transform(
            rasters["mesh_depth"]["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["mesh_depth"]["depth_filled_norm"] = self.depth_transform(
            rasters["mesh_depth"]["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            if self.depth_transform.far_plane_at_max:
                rasters["cuboid_depth"]["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
                rasters["mesh_depth"]["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
            else:
                rasters["cuboid_depth"]["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )
                rasters["mesh_depth"]["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            def resize_all_components(data_dict):
                return {k: resize_transform(v) for k, v in data_dict.items()}
            # Resize all components in the complete_view, cuboid_depth, and mesh_depth
            rasters["complete_view"] = resize_all_components(rasters["complete_view"])
            rasters["cuboid_depth"] = resize_all_components(rasters["cuboid_depth"])
            rasters["mesh_depth"] = resize_all_components(rasters["mesh_depth"])
            rasters["valid_mask_raw"] = resize_transform(rasters["valid_mask_raw"])
            rasters["valid_mask_filled"] = resize_transform(rasters["valid_mask_filled"])
        
        return rasters

