# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import os
import matplotlib.pyplot as plt

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    def __init__(self, rgb_data, depth_data):
        assert rgb_data.shape[0] == depth_data.shape[0], "Mismatched RGB and Depth data size"
        self.rgb_data = rgb_data
        self.depth_data = depth_data

    def __len__(self):
        return len(self.rgb_data)

    def __getitem__(self, idx):
        return self.rgb_data[idx], self.depth_data[idx]


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = 1,
        default_processing_resolution: Optional[int] = 0,
    ):
        super().__init__()
        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image_rgb: Union[Image.Image, torch.Tensor], #camilla
        input_image_depth: Union[Image.Image, torch.Tensor],  # New depth input - camilla
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        

        # ----------------- Image Preprocess -----------------
        # Convert RGB image to torch tensor
        if isinstance(input_image_rgb, Image.Image):
            input_image_rgb = input_image_rgb.convert("RGB")
            input_image_rgb = input_image_rgb.resize((200,200))
            rgb = pil_to_tensor(input_image_rgb).unsqueeze(0)  # [1, rgb, H, W]
        elif isinstance(input_image_rgb, torch.Tensor):
            rgb = input_image_rgb
        else:
            raise TypeError(f"Unknown input type for RGB image: {type(input_image_rgb)}")

        # Convert Depth image to torch tensor
        if isinstance(input_image_depth, Image.Image):
            input_image_depth = input_image_depth.resize((200,200)).convert("L")
            depth = pil_to_tensor(input_image_depth).unsqueeze(0)  # [1, depth, H, W]
        elif isinstance(input_image_depth, torch.Tensor):
            depth = input_image_depth
        else:
            raise TypeError(f"Unknown input type for Depth image: {type(input_image_depth)}")


        print("Depth Shape: ", depth.shape)

        # Ensure both images have the same dimensions
        if rgb.shape[-2:] != depth.shape[-2:]:
            raise ValueError("RGB and Depth images must have the same dimensions.")

        # Resize images if needed
        """
        if processing_res > 0:
            rgb = resize_max_res(rgb, max_edge_resolution=processing_res, resample_method=resample_method)
            depth = resize_max_res(depth, max_edge_resolution=processing_res, resample_method=resample_method)
        """

        # Normalize the RGB values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # Normalize the Depth values (if needed, depending on depth format)
        depth_norm: torch.Tensor = depth / 255.0 * 2.0 - 1.0  # [0, 255] -> [-1, 1]
        depth_norm = depth_norm.to(self.dtype)
        assert depth_norm.min() >= -1.0 and depth_norm.max() <= 1.0



        # ----------------- Predicting cubemap -----------------

        # Expand the input images to create batched data
        duplicated_rgb = rgb_norm.expand(ensemble_size, -1, -1, -1)
        duplicated_depth = depth_norm.expand(ensemble_size, -1, -1, -1)

        # Create the combined dataset
        paired_dataset = PairedDataset(duplicated_rgb, duplicated_depth)

        # Determine batch size
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )
        
        # Create DataLoader for batched processing
        paired_loader = DataLoader(paired_dataset, batch_size=_bs, shuffle=False)
        depth_pred_ls = []
        iterable = tqdm(paired_loader, desc="Inference batches", leave=False) if show_progress_bar else paired_loader

        for rgb_batch, depth_batch in iterable:
            depth_pred_raw = self.single_infer(
                rgb_input=rgb_batch,
                depth_input=depth_batch,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            depth_pred_ls.append(depth_pred_raw.detach())

        depth_preds = torch.cat(depth_pred_ls, dim=0)
        torch.cuda.empty_cache()


        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depth(
                depth_preds,
                scale_invariant=self.scale_invariant,
                shift_invariant=self.shift_invariant,
                max_res=50,
                **(ensemble_kwargs or {}),
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None
        width, height = input_image_rgb.size
        print("Width: ", width)
        # Resize back to original resolution
        if match_input_res:
            depth_pred = resize(
                depth_pred,
                (width, height),
                interpolation=resample_method,
                antialias=True,
            )

        # Convert to numpy
        depth_pred = depth_pred.squeeze()
        depth_pred = depth_pred.cpu().numpy()
        if pred_uncert is not None:
            pred_uncert = pred_uncert.squeeze().cpu().numpy()

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        if color_map is not None:
            depth_colored = colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    

    def single_infer(
            self,
            rgb_input: torch.Tensor,
            depth_input: torch.Tensor,
            num_inference_steps: int,
            generator: Union[torch.Generator, None],
            show_pbar: bool,
            save_plots: bool = True,
            plot_dir: str = "./plots"
        ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoising steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
            save_plots (`bool`, optional):
                Whether to save plots of the original images and their encodings.
            plot_dir (`str`, optional):
                Directory to save plots. Defaults to "./plots".
        Returns:
            `torch.Tensor`: Predicted depth map.
        """

        # ----------------- Predicting depth -----------------
        device = self.device
        rgb_input = rgb_input.to(device)
        depth_input = depth_input.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]
        
        # Encode RGB and Depth inputs
        rgb_latent = self.encode_rgb(rgb_input)  # Encode the RGB input
        depth_latent = self.encode_depth(depth_input)  # Encode the Depth input

        print("RGB Latent Shape: ", rgb_latent.shape)
        print("Depth Latent Shape: ", depth_latent.shape)

        # Save plots if requested
        if save_plots:
            self.save_image_and_encoding(rgb_input, rgb_latent, plot_dir, "rgb")
            self.save_image_and_encoding(depth_input, depth_latent, plot_dir, "depth")

        # Initial depth map (noise)
        depth_latent_noise = torch.randn(
            depth_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, channels, H, W]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (depth_latent.shape[0], 1, 1)
        ).to(device)  # [B, 77, 1024]

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Diffusion denoising", leave=False) if show_pbar else timesteps):
            # Predict the noise residual
            combined_latent = torch.cat([rgb_latent, depth_latent, depth_latent_noise], dim=1)
            noise_pred = self.unet(
                combined_latent, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, channels, H, W]

            # Compute the previous noisy sample x_t -> x_t-1
            depth_latent_noise = self.scheduler.step(
                noise_pred, t, depth_latent_noise, generator=generator
            ).prev_sample

        # Decode the final depth latent
        depth_final = self.decode_depth(depth_latent_noise)

        # Clip prediction and shift to [0, 1]
        depth_final = torch.clip(depth_final, -1.0, 1.0)
        depth_final = (depth_final + 1.0) / 2.0

        return depth_final

    def save_image_and_encoding(self, image, encoding, plot_dir, name):
        """
        Save the original image and its encoding as plots.

        Args:
            image (torch.Tensor): The original input image.
            encoding (torch.Tensor): The encoded representation of the image.
            plot_dir (str): Directory to save the plots.
            name (str): Name prefix for the saved files.
        """

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Convert tensors to numpy arrays for plotting
        image_np = image.squeeze().detach().cpu().numpy()
        encoding_np = encoding.squeeze().detach().cpu().numpy()

        # If the image has 3 channels, move the channels to the last dimension
        if image_np.ndim == 3 and image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        # Plot and save the original image
        plt.figure()
        plt.imshow(image_np, cmap='gray' if image_np.ndim == 2 else None)
        plt.title(f"{name.capitalize()} Input")
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, f"{name}_input.png"))
        plt.close()

        # Plot and save the encoding
        plt.figure()
        plt.imshow(encoding_np[0], cmap='viridis')  # Plot the first channel of encoding
        plt.title(f"{name.capitalize()} Encoding")
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, f"{name}_encoding.png"))
        plt.close()


        

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent
    
    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        logging.info(depth_in.shape)
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean
