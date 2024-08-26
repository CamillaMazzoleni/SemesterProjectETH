import os
from transformers import pipeline as transformers_pipeline
import torch
import diffusers
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class DepthEstimator:
    def __init__(self, model_type="depth-anything", device="cuda"):
        """
        Initialize the DepthEstimator class.

        :param model_type: The type of model to use ("depth-anything" or "marigold").
        :param device: The device to use for computation ("cpu" or "cuda").
        """
        self.model_type = model_type
        self.device = device
        self.pipe = self.load_model()

    def load_model(self):
        """Load the appropriate model based on the model_type."""
        if self.model_type == "depth-anything":
            return transformers_pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if self.device == "cuda" else -1)
        elif self.model_type == "marigold":
            return diffusers.MarigoldDepthPipeline.from_pretrained(
                "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
            ).to(self.device)
        else:
            raise ValueError("Invalid model_type. Choose either 'depth-anything' or 'marigold'.")

    def estimate_depth(self, image_path):
        """Estimate the depth of an image."""
        image = Image.open(image_path)
        
        if self.model_type == "depth-anything":
            depth = self.pipe(image)["depth"]
        elif self.model_type == "marigold":
            depth = self.pipe(image)
        
        return depth

    def visualize_depth(self, depth, save_path=None):
        """Visualize the depth map."""
        if self.model_type == "depth-anything":
            depth_np = np.array(depth)
            depth_np_normalized = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
            plt.figure(figsize=(10, 10))
            plt.imshow(depth_np_normalized, cmap='plasma')
            plt.colorbar(label='Depth')
            plt.title("Depth Map")
            plt.axis('off')
            if save_path:
                plt.savefig(save_path)
            plt.show()
        elif self.model_type == "marigold":
            vis = self.pipe.image_processor.visualize_depth(depth.prediction)
            vis[0].save(save_path if save_path else "depth_map.png")

    def save_depth(self, depth, output_folder, file_name="depth"):
        """Save the depth map."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if self.model_type == "depth-anything":
            depth_image_path = os.path.join(output_folder, f"{file_name}.png")
            depth_image = Image.fromarray((depth / np.max(depth) * 255).astype(np.uint8))
            depth_image.save(depth_image_path)
        elif self.model_type == "marigold":
            depth_16bit = self.pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
            depth_16bit[0].save(os.path.join(output_folder, f"{file_name}_16bit.png"))

    def compare_depths(self, depth1, depth2, title1="Depth 1", title2="Depth 2"):
        """Compare two depth maps side by side."""
        plt.figure(figsize=(20, 10))

        if self.model_type == "depth-anything":
            depth1_np = np.array(depth1)
            depth2_np = np.array(depth2)

            depth1_np_normalized = (depth1_np - np.min(depth1_np)) / (np.max(depth1_np) - np.min(depth1_np))
            depth2_np_normalized = (depth2_np - np.min(depth2_np)) / (np.max(depth2_np) - np.min(depth2_np))

            plt.subplot(1, 2, 1)
            plt.imshow(depth1_np_normalized, cmap='plasma')
            plt.title(title1)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(depth2_np_normalized, cmap='plasma')
            plt.title(title2)
            plt.axis('off')

        elif self.model_type == "marigold":
            vis1 = self.pipe.image_processor.visualize_depth(depth1.prediction)
            vis2 = self.pipe.image_processor.visualize_depth(depth2.prediction)

            plt.subplot(1, 2, 1)
            plt.imshow(vis1[0])
            plt.title(title1)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(vis2[0])
            plt.title(title2)
            plt.axis('off')

        plt.show()