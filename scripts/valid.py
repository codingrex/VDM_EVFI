
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from torch.utils.data import Dataset

from src.dataset_crop_MStack_valid import ValidDataset, NUM_BINS, center_crop

from src.models.unet_spatio_temporal_condition_fullControlnet import UNetSpatioTemporalConditionControlNetModel
from src.pipelines.pipeline_stable_video_diffusion_FullControlnet_MStack_timereversal import StableVideoDiffusionPipelineControlNet
from src.models.fullControlnet_sdv_MStack import ControlNetSDVModel
from diffusers.training_utils import cast_training_params

from src.utils.data_utils import load_event_sequence_npz, get_event_idx_5frame, check_event_sequence_npz
from src.utils.mStack_utils import events_to_voxel_grid, visualize_voxel_grid, voxel_norm, get_event_stacks





NUM_STACKS= NUM_BINS




# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

import pyiqa

mse_loss_fn = nn.MSELoss()

ssim_metric_fn = pyiqa.create_metric('ssim', device='cuda', as_loss=False)
lpips_metric = pyiqa.create_metric('lpips', device='cuda', as_loss=False)









def process_EventImage(events, h, w, num_bins= NUM_BINS, reverse= False):
    """
    Main function
    """


    if len(events) == 0:
        voxel_img= np.zeros((num_bins, h, w))

        save_img_list= visualize_voxel_grid(voxel_img)

        return save_img_list
    
    events_stacks= get_event_stacks(events, NUM_STACKS, reverse= reverse)
    

    events_stack_voxels= []

    for i in range(NUM_STACKS):
        event_voxels_i= events_to_voxel_grid(events_stacks[i], 1, w, h)

        '''
        Normalize the voxel grid first!!!!
        '''

        event_voxels_i= voxel_norm(event_voxels_i)
        
        events_stack_voxels.append(event_voxels_i)

    event_voxels= np.stack(events_stack_voxels, axis=0)

    event_voxels= event_voxels.squeeze(1)

    save_img_list= visualize_voxel_grid(event_voxels)

    return save_img_list


def convert_stack_to_tensor(stack):
    img_tensor = torch.from_numpy(np.array(stack)).float()



    img_tensor_1ch= img_tensor

    # Normalize the image by scaling pixel values to [-1, 1]
    img_normalized = img_tensor_1ch / 127.0 - 1

    img_normalized = img_normalized.unsqueeze(0)


    return img_normalized




def load_image_valid_npy_skip_n(image_path, idx, width, height, num_skips, args= None):
    image_file_path= sorted(os.listdir(image_path))[idx]

    image_file_path= os.path.join(image_path, image_file_path)

    '''
    Load reversed image error: should be idx + num_skips + 1 !!!!!!
    '''
    image_file_path_reversed= sorted(os.listdir(image_path))[idx + num_skips + 1]

    print('image_file_path', image_file_path) 
    print('image_file_path_reversed', image_file_path_reversed)

    image_file_path_reversed= os.path.join(image_path, image_file_path_reversed)


    rescale_factor= args.rescale_factor



    # image_timestamps_file_path= os.path.join(image_path, 'timestamp.txt')

    # if not os.path.exists(image_timestamps_file_path):
    #     image_timestamps_file_path= os.path.join(image_path, '../timestamps.txt')

    # with open(image_timestamps_file_path, 'r') as f:
    #     timestamps= f.readlines()

    # timestamps= [int(float(x.strip())) for x in timestamps]





    valid_image= load_image(image_file_path)

    valid_image_reversed= load_image(image_file_path_reversed)




    events, ev_timestamps= load_event_sequence_npz(valid_image, image_path.replace('images', 'events'), idx, num_frames= num_skips + 2, scale= rescale_factor, event_scale= args.event_scale)


    event_idx_arr= get_event_idx_5frame(events, ev_timestamps, 0, num_skips)

  
    events_arr= []

    events_arr_reversed= []


    # append dummy empty event
    events_arr.append([])

    

    for i in range(len(event_idx_arr) - 1):
        event_i= events[event_idx_arr[i]: event_idx_arr[i + 1]]

        if len(event_i) == 0:
            events_arr.append(event_i)
            events_arr_reversed.append(event_i)
            continue

        event_i= event_i[np.argsort(event_i[:, 0])]

        max_ts_i= np.max(event_i[:, 0])

        events_i_reversed= event_i.copy()

        events_i_reversed[:, 0]= max_ts_i - events_i_reversed[:, 0]

        events_i_reversed= events_i_reversed[np.argsort(events_i_reversed[:, 0])]

        events_arr.append(event_i)

        events_arr_reversed.append(events_i_reversed)

    

    events_arr_reversed.append([])
    
    events_arr_reversed= events_arr_reversed[::-1]


    w, h= valid_image.size


    '''
    Resize to center crop
    '''
    valid_image= valid_image.resize((width, height), resample= Image.LANCZOS)


    valid_image_reversed= valid_image_reversed.resize((width, height), resample= Image.LANCZOS)


    height_upsample = int(height * rescale_factor)
    width_upsample = int(width * rescale_factor)


    ev_pixel_values = torch.empty((num_skips + 2, NUM_BINS, height_upsample, width_upsample))

    ev_pixel_values_reversed = torch.empty((num_skips + 2, NUM_BINS, height_upsample, width_upsample))


    for i in range(len(events_arr)):
        event_i= events_arr[i]
        event_i_reversed= events_arr_reversed[i]


        voxel_img_list= process_EventImage(event_i, h, w)
        voxel_img_reversed_list= process_EventImage(event_i_reversed, h, w, reverse= True)


        for bin in range(NUM_STACKS):
            voxel_img= voxel_img_list[bin]
            voxel_img_reversed= voxel_img_reversed_list[bin]


            voxel_img= voxel_img.resize((width_upsample, height_upsample), resample= Image.LANCZOS)
            voxel_img_reversed= voxel_img_reversed.resize((width_upsample, height_upsample), resample= Image.LANCZOS)



            voxel_img_tensor= convert_stack_to_tensor(voxel_img)
            voxel_img_reversed_tensor= convert_stack_to_tensor(voxel_img_reversed)

            ev_pixel_values[i, bin]= voxel_img_tensor
            ev_pixel_values_reversed[i, bin]= voxel_img_reversed_tensor
            

    return valid_image, ev_pixel_values, valid_image_reversed, ev_pixel_values_reversed



def get_video(folder_path, idx, num_frames, width, height):
    frames = sorted(os.listdir(folder_path))

    # Ensure the selected folder has at least `sample_frames`` frames
    if len(frames) < num_frames:
        raise ValueError(
            f"The selected folder '{folder_path}' contains fewer than `{num_frames}` frames.")

    # Randomly select a start index for frame sequence
    start_idx = idx
    selected_frames = frames[start_idx:start_idx + num_frames]

    video_frames= []

    # Load and process each frame
    for i, frame_name in enumerate(selected_frames):
        frame_path = os.path.join(folder_path, frame_name)
        with Image.open(frame_path) as img:

            if img.mode != 'RGB':
                img = img.convert('RGB')
      
            img_resized = img.resize((width, height))

            video_frames.append(img_resized)

    return video_frames

def calculate_psnr(vid1, vid2):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""

    to_tensor= transforms.ToTensor()

    num_frames = len(vid1)

    psnr_list= []

    for i in range(num_frames):
        img1= vid1[i]
        img2= vid2[i]


        img1= to_tensor(img1)
        img2= to_tensor(img2)

        if mse_loss_fn(img1, img2) == 0:
            psnr_i= 100
        psnr_i= 10*torch.log10(1/mse_loss_fn(img1, img2)).detach().cpu().numpy()

        psnr_list.append(psnr_i)
    

    return np.mean(psnr_list)


def calculate_ssim(vid1, vid2):
    to_tensor= transforms.ToTensor()

    num_frames = len(vid1)

    ssim_list= []

    for i in range(num_frames):
        img1= vid1[i]
        img2= vid2[i]


        img1= to_tensor(img1)
        img2= to_tensor(img2)

        img1= img1.unsqueeze(0)
        img2= img2.unsqueeze(0)

        if i == 0:
            vid1_tensor= img1
            vid2_tensor= img2
        else:
            vid1_tensor= torch.cat((vid1_tensor, img1), dim=0)
            vid2_tensor= torch.cat((vid2_tensor, img2), dim=0)


    ssim_ave= ssim_metric_fn(vid1_tensor, vid2_tensor).detach().cpu().numpy()

        
    

    return np.mean(ssim_ave)

def calculate_lpips(vid1, vid2):
    to_tensor= transforms.ToTensor()

    num_frames = len(vid1)

    lpips_list= []

    for i in range(num_frames):
        img1= vid1[i]
        img2= vid2[i]


        img1= to_tensor(img1)
        img2= to_tensor(img2)


        img1= img1.unsqueeze(0)
        img2= img2.unsqueeze(0)

        if i == 0:
            vid1_tensor= img1
            vid2_tensor= img2
        else:
            vid1_tensor= torch.cat((vid1_tensor, img1), dim=0)
            vid2_tensor= torch.cat((vid2_tensor, img2), dim=0)


    lp_ave= lpips_metric(vid1_tensor, vid2_tensor).detach().cpu().numpy()
    

    return np.mean(lp_ave)



def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)
    


def export_to_images(frames, output_folder_path, part_idx):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    for i, frame in enumerate(pil_frames):
        frame.save(os.path.join(output_folder_path, f'{str(i + part_idx).zfill(6)}.png'))





def convert_to_gif(folder_path, fps):
    frames = sorted(os.listdir(folder_path))
    video_frames = []
    for frame_name in frames:
        frame_path = os.path.join(folder_path, frame_name)
        img= Image.open(frame_path)
        video_frames.append(np.array(img))
    
    output_gif_path = os.path.join(folder_path, "output.gif")

    export_to_gif(video_frames, output_gif_path, fps)
  

def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    # parser.add_argument(
    #     "--base_folder",
    #     required=True,
    #     type=str,
    # )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )

    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--rescale_factor",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--overlapping_ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument("--t0", type=int, default=5, help="Cutoff timestep index for noise injection")
    parser.add_argument("--M", type=int, default=8, help="Number of noise injection steps")
    parser.add_argument("--s_churn", type=float, default=0.5, help="churn tern of scheduler")
    parser.add_argument("--eval_folder_start", type=int, default=0, help="start index of evaluation folder")
    parser.add_argument("--eval_folder_end", type=int, default=1, help="end index of evaluation folder")

    parser.add_argument("--test_data_path", type=str, help="path to test data")


    parser.add_argument("--event_scale", type=int, default=1.0, help="scale for event data")

    parser.add_argument("--video_mode", action="store_true", help="Whether to use video mode.")



    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    args = parse_args()

    SKIP_FRAMES= args.num_frames - 2

    '''
    round args.width and args.height to multiples of 8
    '''
    args.width = args.width - args.width % 8
    args.height = args.height - args.height % 8


    args.output_dir = args.output_dir + f'_rescale_factor_{args.rescale_factor}_overlapping_ratio_{args.overlapping_ratio}_t0_{args.t0}_M_{args.M}_s_churn_{args.s_churn}'

    TEST_DATA_PATH= args.test_data_path



    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")


    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16",
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetSDVModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetSDVModel.from_unet(unet)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_controlnet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))

            

            for i, model in enumerate(models):
                print('model name:', model.__class__.__name__)

                if model.__class__.__name__ == 'ControlNetSDVModel':
                    model.save_pretrained(os.path.join(output_dir, "controlnet"))
                else:
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_controlnet.load_state_dict(load_model.state_dict())
                ema_controlnet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                if model.__class__.__name__ == 'ControlNetSDVModel':
                    # load diffusers style into model
                    load_model = ControlNetSDVModel.from_pretrained(
                        input_dir, subfolder="controlnet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()



    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes


    

    test_dataset = ValidDataset(TEST_DATA_PATH, width=args.width, height=args.height, sample_frames=args.num_frames)

    # Prepare everything with our `accelerator`.
    unet,  test_dataset, controlnet = accelerator.prepare(unet, test_dataset, controlnet
    )

    if args.use_ema:
        ema_controlnet.to(accelerator.device)
        
    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module



    logger.info("***** Running training *****")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(f"rescale factor: {args.rescale_factor}")
    logger.info(f"overlapping ratio: {args.overlapping_ratio}")
    logger.info(f"Number of skip frames: {SKIP_FRAMES}")
    logger.info(f"eval_folder_start: {args.eval_folder_start}")
    logger.info(f"eval_folder_end: {args.eval_folder_end}")
    
  

    # count= 0

    psnr_list= []
    ssim_list= []
    lpips_list= []

    all_folders= sorted(test_dataset.folders)

    # print('all_folders:', len(all_folders))



    '''
    Choose a folder/folders to evaluate
    '''

    eval_folder_start= args.eval_folder_start
    eval_folder_end= args.eval_folder_end

    if eval_folder_end == -1:
        eval_folder_end= len(all_folders)

    folder_idx_arr= np.arange(len(all_folders))[eval_folder_start: eval_folder_end]

    NUM_PARTS_PER_FOLDER= 4


    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                image_encoder=accelerator.unwrap_model(
                    image_encoder),
                controlnet=accelerator.unwrap_model(controlnet),
                vae=accelerator.unwrap_model(vae),
                revision=args.revision,
                torch_dtype=weight_dtype,
            )
    pipeline = pipeline.to(accelerator.device)

    # Save GPU memory by cpu offloading the model
    pipeline.enable_model_cpu_offload()
    # pipe.unet.enable_forward_chunking()


    val_epoch_dir= None

    for epoch in tqdm(folder_idx_arr):
        folder_eval= test_dataset.folders[epoch]

        print('folder_eval:', folder_eval)

        count= 0


        # number of png files in the folder
        num_files= len(os.listdir(os.path.join(TEST_DATA_PATH, folder_eval, 'images'))) - 1


        NUM_PARTS_PER_FOLDER= 1
        controlnet.eval()
        for i in range(0, max(1, num_files - (args.num_frames + 3)), SKIP_FRAMES + 1):
            print('i:', i)
            batch= test_dataset[epoch, i, NUM_PARTS_PER_FOLDER, SKIP_FRAMES]

            
            pipeline.set_progress_bar_config(disable=True)

            val_save_dir = os.path.join(
                    args.output_dir, "test_images")

            if not os.path.exists(val_save_dir):
                os.makedirs(val_save_dir)

            out_file = os.path.join(val_save_dir,f"step_{str(count).zfill(4)}.mp4")

            gt_save_dir = os.path.join(
                args.output_dir, "gt_test_images")
            
            if not os.path.exists(gt_save_dir):
                os.makedirs(gt_save_dir)

            gt_out_file = os.path.join(gt_save_dir,f"step_{str(count).zfill(4)}.mp4")


            image_folder_path= batch['rgb_folder_path']
            idx= batch['start_idx']

            print('start idx:', idx)

            # test_img= cv2.imread(os.path.join(image_folder_path, f'{str(idx).zfill(6)}.png'))

            test_img= cv2.imread(os.path.join(image_folder_path, sorted(os.listdir(image_folder_path))[0]))

            if check_event_sequence_npz(test_img, image_folder_path.replace('images', 'events'), i, args.num_frames, scale=1) == False:
                print('Skipping', i)
                count+= 1
                continue

            valid_image, valid_ev_imgs, valid_image_rev, valid_ev_imgs_rev =load_image_valid_npy_skip_n(image_folder_path, idx, args.width, args.height, SKIP_FRAMES ,args= args)


            num_frames = args.num_frames


            valid_videos= get_video(image_folder_path, idx, num_frames, args.width, args.height)


            org_frames= valid_videos.copy()


            if args.video_mode:
                org_frames= None


            with torch.autocast(
                                    str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                                ):
                

            

                video_frames, org_frames = pipeline(
                    valid_image,
                    valid_ev_imgs,
                    valid_image_rev,
                    valid_ev_imgs_rev,
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    decode_chunk_size=args.decode_chunk_size,
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                    rescale_factor= args.rescale_factor,
                    num_inference_steps= args.num_inference_steps,
                    # generator=generator,
                    overlap_ratio= args.overlapping_ratio, 
                    t0= args.t0,
                    M= args.M,
                    s_churn= args.s_churn,
                    org_frames= org_frames
                )

                video_frames= video_frames.frames[0]


                if not args.video_mode:
                    org_frames= org_frames[0]


                valid_videos= np.array(valid_videos)
                video_frames= np.array(video_frames)


                if not args.video_mode:
                    original_video_frames= np.array(org_frames)



            resized_valid_frames= []



            if video_frames[0].shape != valid_videos[0].shape:
                for k in range(len(video_frames)):
                    resized_valid_frame= Image.fromarray(video_frames[k])

                    resized_valid_frame= resized_valid_frame.resize((args.width, args.height), Image.LANCZOS)

                    resized_valid_frame= np.array(resized_valid_frame)

                    resized_valid_frames.append(resized_valid_frame)

                video_frames= np.array(resized_valid_frames)

            

            if not args.video_mode:
                resized_valid_frames= []
                if original_video_frames[0].shape != valid_videos[0].shape:
                    for k in range(len(original_video_frames)):
                        resized_valid_frame= Image.fromarray(original_video_frames[k])

                        resized_valid_frame= resized_valid_frame.resize((args.width, args.height), Image.LANCZOS)

                        resized_valid_frame= np.array(resized_valid_frame)

                        resized_valid_frames.append(resized_valid_frame)

                    original_video_frames= np.array(resized_valid_frames)


            val_epoch_dir= os.path.join(val_save_dir, f'{folder_eval}')

            gt_epoch_dir= os.path.join(gt_save_dir, f'{folder_eval}')

            org_epoch_dir= os.path.join(gt_save_dir, f'{folder_eval}')

            if not os.path.exists(val_epoch_dir):
                os.makedirs(val_epoch_dir)


            if not os.path.exists(gt_epoch_dir):
                os.makedirs(gt_epoch_dir)


            if not os.path.exists(org_epoch_dir):
                os.makedirs(org_epoch_dir)

           

            export_to_images(video_frames, val_epoch_dir, i)

            export_to_images(valid_videos, gt_epoch_dir, i)


            if not args.video_mode:
                export_to_images(original_video_frames, org_epoch_dir, i)


            


            valid_videos= valid_videos[1:-1]
            video_frames= video_frames[1:-1]


            if not args.video_mode:

                original_video_frames= original_video_frames[1:-1]
                psnr_org= calculate_psnr(original_video_frames, video_frames)
                ssim_org= calculate_ssim(original_video_frames, video_frames)
                lpips_org= calculate_lpips(original_video_frames, video_frames)

                psnr_list.append(psnr_org)

                print('psnr_org:', psnr_org)

                ssim_list.append(ssim_org)

                lpips_list.append(lpips_org)

        

                save_metrics_folder_psnr= os.path.join(args.output_dir, 'metrics', 'PSNR', f'{folder_eval}')

                save_metrics_folder_ssim= os.path.join(args.output_dir, 'metrics', 'SSIM', f'{folder_eval}')

                save_metrics_folder_lpips= os.path.join(args.output_dir, 'metrics', 'LPIPS', f'{folder_eval}')

                if not os.path.exists(save_metrics_folder_psnr):
                    os.makedirs(save_metrics_folder_psnr)
                
                if not os.path.exists(save_metrics_folder_ssim):
                    os.makedirs(save_metrics_folder_ssim)
                
                if not os.path.exists(save_metrics_folder_lpips):
                    os.makedirs(save_metrics_folder_lpips)
                


                save_metrics_PSNR_txt= open(os.path.join(save_metrics_folder_psnr, f'step_{str(i).zfill(4)}.txt'), 'w')
                save_metrics_SSIM_txt= open(os.path.join(save_metrics_folder_ssim, f'step_{str(i).zfill(4)}.txt'), 'w')
                save_metrics_LPIPS_txt= open(os.path.join(save_metrics_folder_lpips, f'step_{str(i).zfill(4)}.txt'), 'w')

                print(f'{psnr_org}', file=save_metrics_PSNR_txt)
                print(f'{ssim_org}', file=save_metrics_SSIM_txt)
                print(f'{lpips_org}', file=save_metrics_LPIPS_txt)

                save_metrics_PSNR_txt.close()
                save_metrics_SSIM_txt.close()
                save_metrics_LPIPS_txt.close()


            count+= 1


    mean_psnr= np.mean(psnr_list)
    mean_ssim= np.mean(ssim_list)
    mean_lpips= np.mean(lpips_list)

    print('mean psnr:', mean_psnr)
    print('mean ssim:', mean_ssim)
    print('mean lpips:', mean_lpips)


    del pipeline
    torch.cuda.empty_cache()




if __name__ == "__main__":
    main()
