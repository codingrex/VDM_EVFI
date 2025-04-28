from torch.utils.data import Dataset

import os

import random

import torch

import numpy as np

from PIL import Image

import cv2

from diffusers.utils import load_image


CROP_x= 512
CROP_y= 320


NUM_BINS= 6


def get_center_crop_coords(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    left = int(left)
    top = int(top)

    return left, top





def get_views(panorama_height, panorama_width, overlap_ratio=0.5):
    
    panorama_height /= 8
    panorama_width /= 8

    print('panorama_height', panorama_height)
    print('panorama_width', panorama_width)

    window_size_x= CROP_x // 8
    window_size_y= CROP_y // 8


    stride_x = int(window_size_x * (1 - overlap_ratio))
    stride_y = int(window_size_y * (1 - overlap_ratio))


    # num_blocks_height = (panorama_height - window_size_y) // stride_y + 1
    # num_blocks_width = (panorama_width - window_size_x) // stride_x + 1

    # account for residual blocks
    num_blocks_height = (panorama_height - window_size_y) // stride_y + 1
    if (panorama_height - window_size_y) % stride_y != 0:
        num_blocks_height += 1 
    
    num_blocks_width = (panorama_width - window_size_x) // stride_x + 1
    if (panorama_width - window_size_x) % stride_x != 0:
        num_blocks_width += 1



    total_num_blocks = int(num_blocks_height * num_blocks_width)

    views = []

    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride_y)
        h_end = h_start + window_size_y
        if h_end > panorama_height:
            h_end = panorama_height
            h_start = h_end - window_size_y
        w_start = int((i % num_blocks_width) * stride_x)
        w_end = w_start + window_size_x
        if w_end > panorama_width:
            w_end = panorama_width
            w_start = w_end - window_size_x
        
        views.append((w_start, h_start, w_end, h_end))


    return views


def apply_crop(image, start_x, start_y, crop_size_x, crop_size_y):
    image_croped = image.crop((start_x, start_y, start_x + crop_size_x, start_y + crop_size_y))

    return image_croped  


def get_random_crop_idx(image, crop_size_x, crop_size_y):
    width, height = image.size

    start_x = random.randint(0, width - crop_size_x)
    start_y = random.randint(0, height - crop_size_y)

    return start_x, start_y

def divide_image(image, crop_size_x, crop_size_y):
    width, height = image.size

    num_parts_x = width // crop_size_x
    num_parts_y = height // crop_size_y
    
    residual_x = width % crop_size_x
    residual_y = height % crop_size_y

    image_parts = []

    for i in range(num_parts_x):
        for j in range(num_parts_y):
            start_x = i * crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0:
        for j in range(num_parts_y):
            start_x = width - crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_y > 0:
        for i in range(num_parts_x):
            start_x = i * crop_size_x
            start_y = height - crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0 and residual_y > 0:
        start_x = width - crop_size_x
        start_y = height - crop_size_y

        image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

        image_parts.append(image_croped)


    return image_parts


def get_image_parts(image, crop_size_x, crop_size_y, idx):
    width, height = image.size

    num_parts_x = width // crop_size_x
    num_parts_y = height // crop_size_y
    
    residual_x = width % crop_size_x
    residual_y = height % crop_size_y

    image_parts = []

    for i in range(num_parts_x):
        for j in range(num_parts_y):
            start_x = i * crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0:
        for j in range(num_parts_y):
            start_x = width - crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_y > 0:
        for i in range(num_parts_x):
            start_x = i * crop_size_x
            start_y = height - crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0 and residual_y > 0:
        start_x = width - crop_size_x
        start_y = height - crop_size_y

        image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

        image_parts.append(image_croped)


    return image_parts[idx]




def check_all_files_exist(file_list, folder_path):

    return all([os.path.isfile(os.path.join(folder_path, file)) for file in file_list])



def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img.crop((left, top, right, bottom))



class ValidDataset(Dataset):
    def __init__(self, base_folder, num_samples=100000, width=1024, height=576, sample_frames=14):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples

        self.base_folder = base_folder
        self.folders = sorted(os.listdir(self.base_folder))
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx= 0):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
   

        folder_idx= idx[0]
        frame_idx= idx[1]
        nparts= idx[2]

        skip= idx[3]

        # slecet a folder by index
        chosen_folder = self.folders[folder_idx]
        folder_path = os.path.join(self.base_folder, chosen_folder)


        # Get from rgb folder
        rgb_folder_path = os.path.join(folder_path, 'images')
        
        frames = os.listdir(rgb_folder_path)
        # Sort the frames by name
        frames.sort()

        
        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")



        start_idx = frame_idx



        selected_frames = frames[start_idx:start_idx + self.sample_frames]


        while check_all_files_exist(selected_frames, rgb_folder_path) == False:
            start_idx = random.randint(0, len(frames) - self.sample_frames)
            selected_frames = frames[start_idx:start_idx + self.sample_frames]





        return {'rgb_folder_path': rgb_folder_path, 'start_idx': start_idx}