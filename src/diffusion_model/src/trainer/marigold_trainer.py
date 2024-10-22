# An official reimplemented version of Marigold training script.
# Last modified: 2024-04-29
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import binary_dilation

from marigold.marigold_pipeline import MarigoldPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence
import wandb

import open3d as o3d

import numpy as np
import torch
torch.cuda.empty_cache()


def tensor_to_image(tensor):
    tensor = tensor.squeeze().cpu().detach().numpy()
    #tensor = (tensor + 1.0) / 2.0
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
    else:
        tensor = tensor[0]
    return tensor


class MarigoldTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.depth_mask = False
        self.cfg: OmegaConf = cfg
        self.model: MarigoldPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Adapt input layers
        if 12 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in()

        logging.info(self.val_loaders)

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)

        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
                
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        print(self.prediction_type)
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming

    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 3, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.33
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            12, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return
    """
    def _dilate_and_downsample_mask(self, invalid_mask):
        # Step 1: Apply dilation to each 2D depth slice
        print("Original mask shape:", invalid_mask.shape)  # Print original mask shape

        dilated_valid_mask_list = []
        for i in range(invalid_mask.shape[0]):  # Iterate over the batch dimension
            single_mask = invalid_mask[i, 0].cpu().numpy()  # Get [H, W] mask
            dilated_single_mask = binary_dilation(single_mask, structure=np.ones((3, 3)))
            dilated_valid_mask_list.append(torch.tensor(dilated_single_mask, device=self.device))
        
        dilated_valid_mask = torch.stack(dilated_valid_mask_list).unsqueeze(1)
        print("After dilation, mask shape:", dilated_valid_mask.shape)  # Print shape after dilation
        
        # Step 2: Downsample the mask to match latent tensor size
        dilated_valid_mask = ~torch.max_pool2d(dilated_valid_mask.float(), 8, 8).bool()
        print("After downsampling, mask shape:", dilated_valid_mask.shape)  # Print shape after downsampling
        
        # Repeat mask to match latent channels
        dilated_valid_mask = dilated_valid_mask.repeat((1, 4, 1, 1))
        print("After repeating to match latent channels, mask shape:", dilated_valid_mask.shape)  # Print shape after repeating

        return dilated_valid_mask
    """


    def train(self, t_end=None):
        logging.info("Start training")
        logging.info(f"Loss type: {self.prediction_type}")
        
        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0
        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):

                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # Get data
                rgb = batch['complete_view']['rgb_norm'].to(device)  # RGB input, shape [B, 3, H, W]
                depth_input = batch['cuboid_depth'][self.gt_depth_type].to(device)
                depth_output = batch['mesh_depth'][self.gt_depth_type].to(device)

                #depth_input = 1 - depth_input
                #depth_output = 1 - depth_output
                # Create valid masks for both depth inputs and outputs
                #valid_mask_raw = batch["valid_mask_raw"].to(device)  # From ShapeNetDataset
                #valid_mask_filled = batch["valid_mask_filled"].to(device)


                if self.gt_mask_type is not None:
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                else:
                    raise NotImplementedError

                # Apply depth masks to input and output depths
                """
                if self.depth_mask:
                    # Create an invalid mask (invert valid mask)
                    invalid_mask = ~valid_mask_raw
                   
                    dilated_valid_mask = self._dilate_and_downsample_mask(invalid_mask)

                    # Apply the mask to the depth latents
                    depth_input = depth_input[dilated_valid_mask]
                    depth_output = depth_output[dilated_valid_mask]
                    # Step 3: Check the shapes and values of the mask to ensure correctness
                    print(f"Depth Output Shape: {depth_output.shape}")
                    print(f"Valid Mask Shape: {valid_mask_raw.shape}")
                    print(f"Dilated Mask Shape: {dilated_valid_mask_raw.shape}")
                    print(f"Valid Mask (before dilation) sample: {valid_mask[0, 0, :5, :5]}")
                    print(f"Dilated Mask sample: {dilated_valid_mask[0, 0, :5, :5]}")
                

                
                if self.depth_mask:
                    invalid_mask = ~valid_mask_for_latent  # Invert the valid mask
                    
                    # Step 2: Apply dilation to each 2D depth slice
                    dilated_valid_mask_list = []
                    for i in range(invalid_mask.shape[0]):  # Iterate over the batch dimension
                        # Extract each mask for dilation
                        single_mask = invalid_mask[i, 0].cpu().numpy()  # Get [H, W] mask
                        
                        # Apply binary dilation to the 2D mask
                        dilated_single_mask = binary_dilation(single_mask, structure=np.ones((3, 3)))
                        
                        # Convert back to tensor and append to the list
                        dilated_valid_mask_list.append(torch.tensor(dilated_single_mask, device=device))

                    # Stack the dilated masks back into a 4D tensor [B, 1, H, W]
                    dilated_valid_mask = torch.stack(dilated_valid_mask_list).unsqueeze(1)
                    
                    # Step 3: Downsample the dilated mask to match the latent tensor size
                    dilated_valid_mask = ~torch.max_pool2d(
                        dilated_valid_mask.float(), 8, 8
                    ).bool()  # Downsample with kernel and stride of 8

                    # Step 4: Repeat the downsampled mask to match the number of channels (4 in your case)
                    dilated_valid_mask = dilated_valid_mask.repeat((1, 4, 1, 1))

                    # Step 3: Check the shapes and values of the mask to ensure correctness
                    print(f"Depth Output Shape: {depth_output.shape}")
                    print(f"Valid Mask Shape: {valid_mask.shape}")
                    print(f"Dilated Mask Shape: {dilated_valid_mask.shape}")
                    print(f"Valid Mask (before dilation) sample: {valid_mask[0, 0, :5, :5]}")
                    print(f"Dilated Mask sample: {dilated_valid_mask[0, 0, :5, :5]}")
                """
                
                rgb_image = rgb[0].cpu().numpy().transpose(1, 2, 0)
                depth_input_image = depth_input[0].cpu().numpy()[0]  
                depth_output_image = depth_output[0].cpu().numpy()[0]  
                depth_input_image = (depth_input_image - depth_input_image.min()) / (depth_input_image.max() - depth_input_image.min())
                depth_output_image = (depth_output_image - depth_output_image.min()) / (depth_output_image.max() - depth_output_image.min())
                
                
                # Log images to W&B
                wandb.log({
                    "RGB Image": wandb.Image(rgb_image),
                    "Depth Input": wandb.Image(depth_input_image),
                    "Depth Output": wandb.Image(depth_output_image)
                })
         
                #TODO probably have to change this
                """
                if self.gt_mask_type is not None:
                    valid_mask_for_latent = batch[self.gt_mask_type].to(device)
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down = ~torch.max_pool2d(
                        invalid_mask.float(), 8, 8
                    ).bool()
                    valid_mask_down = valid_mask_down.repeat((1, 4, 1, 1))
                else:
                    raise NotImplementedError
                """

                batch_size = rgb.shape[0]
                

                with torch.no_grad():
                    # Encode image
                    rgb_latent = self.model.encode_rgb(rgb)  # [B, 4, h, w]
                    input_depth_latent = self.encode_depth(depth_input)
                    output_depth_latent = self.encode_depth(depth_output)


                    """

                    encode_try = depth_output.squeeze()
                    encode_try = encode_try.cpu().numpy()
                    img_name = "try_encode.png"
                    encode_try = (encode_try + 1.0) / 2.0
                    depth_to_save = (encode_try * 65535.0).astype(np.uint16)
                    Image.fromarray(depth_to_save).save(img_name, mode="I;16")
                    """

                    # Decode the depth latent
                    input_try = self.model.decode_depth(input_depth_latent)

                    # Squeeze and convert to numpy
                    input_try = input_try.squeeze().cpu().numpy()

                    # Print min and max values after squeezing
                    print(f"After squeezing: min={input_try.min()}, max={input_try.max()}")

                    # Normalize to range [0, 1]
                    input_try = (input_try + 1.0) / 2.0

                    # Print min and max values after normalization
                    print(f"After normalization: min={input_try.min()}, max={input_try.max()}")

                    # 1. Multiply by 1000 and clip
                    input_1 = input_try * 1000  # Convert to millimeters
                    input_1_clipped = np.clip(input_1, 0, 65535).astype(np.uint16)
                    img_name_1 = "try_decode_input_mult_clip.png"
                    new_depth_image_1 = o3d.geometry.Image(input_1_clipped)
                    o3d.io.write_image(img_name_1, new_depth_image_1)

                    # 2. Multiply by 1000 and don't clip
                    input_2 = input_try * 1000  # Convert to millimeters without clipping
                    img_name_2 = "try_decode_input_mult_noclip.png"
                    new_depth_image_2 = o3d.geometry.Image(input_2.astype(np.uint16))  # This might overflow
                    o3d.io.write_image(img_name_2, new_depth_image_2)

                    # 3. No multiplication, but clip
                    input_3_clipped = np.clip(input_try, 0, 65535).astype(np.uint16)  # No multiplication, clip values
                    img_name_3 = "try_decode_input_nomult_clip.png"
                    new_depth_image_3 = o3d.geometry.Image(input_3_clipped)
                    o3d.io.write_image(img_name_3, new_depth_image_3)

                    # 4. No multiplication and no clipping
                    input_4 = input_try  # No multiplication and no clipping
                    img_name_4 = "try_decode_input_nomult_noclip.png"
                    new_depth_image_4 = o3d.geometry.Image(input_4.astype(np.uint16))  # This might cause range issues
                    o3d.io.write_image(img_name_4, new_depth_image_4)

                    print("All four depth images have been saved.")

                    """
                    decode_try = self.model.decode_depth(output_depth_latent)
                    decode_try = torch.clip(decode_try, -1.0, 1.0)
                    decode_try = (decode_try + 1.0) / 2.0
                    decode_try = decode_try.squeeze()
                    decode_try = decode_try.cpu().numpy()
                    img_name = "try_decode.png"
                    depth_to_save = (decode_try * 65535.0).astype(np.uint16)
                    Image.fromarray(depth_to_save).save(img_name, mode="I;16")
                    """

                    if self.depth_mask:
                        input_depth_latent = input_depth_latent[dilated_valid_mask]
                        output_depth_latent = output_depth_latent[dilated_valid_mask]
                    

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.scheduler_timesteps,
                    (batch_size,),
                    device=device,
                    generator=rand_num_generator,
                ).long()  # [B]

                # Sample noise
                if self.apply_multi_res_noise:
                    strength = self.mr_noise_strength
                    if self.annealed_mr_noise:
                        # calculate strength depending on t
                        strength = strength * (timesteps / self.scheduler_timesteps)
                    noise = multi_res_noise_like(
                        output_depth_latent,
                        strength=strength,
                        downscale_strategy=self.mr_noise_downscale_strategy,
                        generator=rand_num_generator,
                        device=device,
                    )
                else:
                    noise = torch.randn(
                        output_depth_latent.shape,
                        device=device,
                        generator=rand_num_generator,
                    )  # [B, 4, h, w]

                # Add noise to the latents (diffusion forward process)
                noisy_latents = self.training_noise_scheduler.add_noise(
                    output_depth_latent, noise, timesteps
                )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                # Concat rgb and depth latents
                cat_latents = torch.cat(
                    [rgb_latent, input_depth_latent, noisy_latents], dim=1
                )  # [B, 8, h, w]
                cat_latents = cat_latents.float()

                # Predict the noise residual
                model_pred = self.model.unet(
                    cat_latents, timesteps, text_embed
                ).sample  # [B, 4, h, w]
                if torch.isnan(model_pred).any():
                    logging.warning("model_pred contains NaN.")

                # Get the target for loss depending on the prediction type
                if "sample" == self.prediction_type:
                    target = output_depth_latent
                elif "epsilon" == self.prediction_type:
                    target = noise
                elif "v_prediction" == self.prediction_type:
                    target = self.training_noise_scheduler.get_velocity(
                        output_depth_latent, noise, timesteps
                    )  # [B, 4, h, w]
                elif "difference" == self.prediction_type:
                    #target = input_depth_latent + model_pred
                    target = output_depth_latent - input_depth_latent

                else:
                    raise ValueError(f"Unknown prediction type {self.prediction_type}")

                # Masked latent loss
                
                    

                encoded_rgb_images = []
                encoded_depth_input_images = []
                encoded_depth_output_images = []
                predicted_images = []

                for i in range(rgb_latent.shape[0]):
                    encoded_rgb_image = tensor_to_image(rgb_latent[i])
                    encoded_depth_input_image = tensor_to_image(input_depth_latent[i])
                    encoded_depth_output_image = tensor_to_image(output_depth_latent[i])
                    predicted_image = tensor_to_image(model_pred[i])

                    # Add images to the lists for logging later
                    encoded_rgb_images.append(wandb.Image(encoded_rgb_image, caption=f"Encoded RGB {i}"))
                    encoded_depth_input_images.append(wandb.Image(encoded_depth_input_image, caption=f"Encoded Depth Input {i}"))
                    encoded_depth_output_images.append(wandb.Image(encoded_depth_output_image, caption=f"Encoded Depth Output {i}"))
                    predicted_images.append(wandb.Image(predicted_image, caption=f"Predicted Output {i}"))

                wandb.log({
                    "Encoded RGB": encoded_rgb_images,
                    "Encoded Depth Input Images": encoded_depth_input_images,
                    "Encoded Depth Output Images": encoded_depth_output_images,
                    "Predicted Images": predicted_images,
                })

                # Masked latent loss
                if self.gt_mask_type is not None:
                    latent_loss = self.loss(
                        model_pred[valid_mask_down].float(),
                        target[valid_mask_down].float(),
                    )
                else:
                    latent_loss = self.loss(model_pred.float(), target.float())
                
                loss = latent_loss.mean()

                self.train_metrics.update("loss", loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0

    def encode_depth(self, depth_in):
        # stack depth into 3-channel
        stacked = self.stack_depth_images(depth_in)
        # encode using VAE encoder
        depth_latent = self.model.encode_rgb(stacked)
        return depth_latent

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    def validate(self):
        logging.info("validation phase")
        for i, val_loader in enumerate(self.val_loaders):
            #val_dataset_name = val_loader.dataset.dispname
            val_dataset_name ="shapenet"
            val_loader_filename = "single_image.json"

            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader_filename,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    def visualize(self):
        logging.info("visualization phase")
        for val_loader in self.vis_loaders:
            #vis_dataset_name = val_loader.dataset.dispname
            vis_dataset_name = "shapenet"
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            print(f"saving at {vis_out_dir}")
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )

    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on dataset"),
            start=1,
        ):
            assert 1 == data_loader.batch_size
            
            # Read input images
            rgb_int = batch['complete_view']['rgb_int']
            depth_input = batch['cuboid_depth']['depth_raw_linear'].squeeze() #tocheck
            depth_raw_ts = batch["mesh_depth"]['depth_raw_linear'].squeeze() #tocheck
            depth_raw = depth_raw_ts.numpy()
            depth_raw_ts = depth_raw_ts.to(self.device)
            valid_mask_ts = batch["valid_mask_raw"].squeeze()
            print(valid_mask_ts.shape)
            valid_mask = valid_mask_ts.numpy()
            valid_mask_ts = valid_mask_ts.to(self.device)
            
            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out: MarigoldDepthOutput = self.model(
                rgb_int,
                depth_input,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            depth_pred: np.ndarray = pipe_out.depth_np

            if "least_square" == self.cfg.eval.alignment:
                depth_pred, scale, shift = align_depth_least_square(
                    gt_arr=depth_raw,
                    pred_arr=depth_pred,
                    valid_mask_arr=valid_mask,
                    return_scale_shift=True,
                    max_resolution=self.cfg.eval.align_max_res,
                )
            else:
                raise RuntimeError(f"Unknown alignment type: {self.cfg.eval.alignment}")

            # Clip to dataset min max
            depth_pred = np.clip(
                depth_pred,
                a_min=data_loader.dataset.min_depth,
                a_max=data_loader.dataset.max_depth,
            )
            print(depth_pred.min())
            print(depth_pred.max())

            # clip to d > 0 for evaluation
            depth_pred = np.clip(depth_pred, a_min=1e-6, a_max=None)

            # Evaluate
            sample_metric = []
            depth_pred_ts = torch.from_numpy(depth_pred).to(self.device)

            for met_func in self.metric_funcs:
                _metric_name = met_func.__name__
                _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                sample_metric.append(_metric.__str__())
                metric_tracker.update(_metric_name, _metric)

            # Save as 16-bit uint png
            if save_to_dir is not None:
                img_name = batch["path_to_image"][0].replace("/", "_")
                png_save_path = os.path.join(save_to_dir, f"{img_name}.png")
                depth_to_save = (pipe_out.depth_np * 65535.0).astype(np.uint16)
                Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

        return metric_tracker.result()

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"